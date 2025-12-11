// Vercel Serverless Function
// AtomS3R から送られてきた Base64 PCM(16kHz/16bit/mono) を
// 1. WAV に変換して Azure Speech STT へ送信
// 2. 得られたテキストを Azure OpenAI Chat Completions に投げる
// 3. 返答テキストを Azure Speech TTS に渡して WAV(PCM) を生成
// 4. WAV バイナリをそのままレスポンスとして返す

/**
 * PCM16 (16kHz, mono) Buffer から WAV Buffer を生成
 * @param {Buffer} pcmBuffer
 * @param {Object} options
 * @param {number} options.sampleRate
 * @param {number} options.numChannels
 * @returns {Buffer}
 */
function pcm16ToWavBuffer(pcmBuffer, { sampleRate = 16000, numChannels = 1 } = {}) {
  const bitsPerSample = 16;
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const wavHeaderSize = 44;
  const dataSize = pcmBuffer.length;

  const buffer = Buffer.alloc(wavHeaderSize + dataSize);

  // RIFF ヘッダ
  buffer.write('RIFF', 0); // ChunkID
  buffer.writeUInt32LE(36 + dataSize, 4); // ChunkSize = 36 + Subchunk2Size
  buffer.write('WAVE', 8); // Format

  // fmt チャンク
  buffer.write('fmt ', 12); // Subchunk1ID
  buffer.writeUInt32LE(16, 16); // Subchunk1Size (16 for PCM)
  buffer.writeUInt16LE(1, 20); // AudioFormat (1 = PCM)
  buffer.writeUInt16LE(numChannels, 22); // NumChannels
  buffer.writeUInt32LE(sampleRate, 24); // SampleRate
  buffer.writeUInt32LE(byteRate, 28); // ByteRate
  buffer.writeUInt16LE(blockAlign, 32); // BlockAlign
  buffer.writeUInt16LE(bitsPerSample, 34); // BitsPerSample

  // data チャンク
  buffer.write('data', 36); // Subchunk2ID
  buffer.writeUInt32LE(dataSize, 40); // Subchunk2Size

  // PCM データ本体
  pcmBuffer.copy(buffer, wavHeaderSize);

  return buffer;
}

function escapeForXml(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

async function callAzureSpeechToText(wavBuffer) {
  const speechKey = process.env.AZURE_SPEECH_KEY;
  const speechRegion = process.env.AZURE_SPEECH_REGION;

  if (!speechKey || !speechRegion) {
    throw new Error('AZURE_SPEECH_KEY / AZURE_SPEECH_REGION が設定されていません。');
  }

  const url = `https://${speechRegion}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=ja-JP`;

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Ocp-Apim-Subscription-Key': speechKey,
      'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000',
      'Accept': 'application/json'
    },
    body: wavBuffer
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Azure Speech STT でエラーが発生しました: ${res.status} ${res.statusText} ${text}`);
  }

  const data = await res.json();
  // REST API は RecognitionStatus / DisplayText などを返す
  let recognizedText = data.DisplayText || data.Text || '';

  if (!recognizedText) {
    // throw new Error('音声認識結果が空でした。');
    recognizedText = 'エラーが発生しました。';
  }

  return recognizedText;
}

async function callAzureOpenAIChat(promptText) {
  const openaiKey = process.env.AZURE_OPENAI_KEY;
  const openaiEndpoint = process.env.AZURE_OPENAI_ENDPOINT;
  const openaiDeployment = process.env.AZURE_OPENAI_DEPLOYMENT;
  const apiVersion = process.env.AZURE_OPENAI_API_VERSION || '2024-02-15-preview';

  if (!openaiKey || !openaiEndpoint || !openaiDeployment) {
    throw new Error('AZURE_OPENAI_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT が設定されていません。');
  }

  const endpoint = openaiEndpoint.replace(/\/$/, '');
  const url = `${endpoint}/openai/deployments/${openaiDeployment}/chat/completions?api-version=${apiVersion}`;

  const body = {
    messages: [
      {
        role: 'system',
        content: 'あなたは親切な日本語の会話アシスタントです。できるだけ簡潔に、わかりやすく答えてください。'
      },
      {
        role: 'user',
        content: promptText
      }
    ],
    max_tokens: 256,
    temperature: 0.7
  };

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'api-key': openaiKey,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body)
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Azure OpenAI Chat でエラーが発生しました: ${res.status} ${res.statusText} ${text}`);
  }

  const data = await res.json();
  const replyText =
    data.choices?.[0]?.message?.content?.trim() ||
    data.choices?.[0]?.messages?.[0]?.content?.trim() ||
    '';

  if (!replyText) {
    throw new Error('Azure OpenAI からの返答が取得できませんでした。');
  }

  return replyText;
}

async function callAzureSpeechTTS(text) {
  const speechKey = process.env.AZURE_SPEECH_KEY;
  const speechRegion = process.env.AZURE_SPEECH_REGION;

  if (!speechKey || !speechRegion) {
    throw new Error('AZURE_SPEECH_KEY / AZURE_SPEECH_REGION が設定されていません。');
  }

  const url = `https://${speechRegion}.tts.speech.microsoft.com/cognitiveservices/v1`;

  // 日本語の女性音声例: ja-JP-NanamiNeural
  const ssml = `
<speak version="1.0" xml:lang="ja-JP">
  <voice xml:lang="ja-JP" xml:gender="Female" name="ja-JP-NanamiNeural">
    ${escapeForXml(text)}
  </voice>
</speak>`;

  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Ocp-Apim-Subscription-Key': speechKey,
      'Content-Type': 'application/ssml+xml',
      // WAV (RIFF) 形式で 16kHz/16bit/mono の PCM を取得
      // → デバイス側では先頭 44 バイトのヘッダをスキップして生 PCM として再生できる
      'X-Microsoft-OutputFormat': 'riff-16khz-16bit-mono-pcm',
      'User-Agent': 'pocket-ai-vercel'
    },
    body: ssml
  });

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`Azure Speech TTS でエラーが発生しました: ${res.status} ${res.statusText} ${text}`);
  }

  const arrayBuffer = await res.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method Not Allowed. Use POST.' });
    return;
  }

  try {
    const body = typeof req.body === 'string' ? JSON.parse(req.body || '{}') : (req.body || {});

    // AtomS3R 側の JSON 仕様:
    // { "audioBase64": "<PCM 16kHz/16bit/mono の Base64 文字列>" }
    const base64 = body.audioBase64 || body.audio || '';
    if (!base64 || typeof base64 !== 'string') {
      res.status(400).json({ error: 'audioBase64 (Base64 PCM) が指定されていません。' });
      return;
    }

    // Base64 → PCM Buffer
    const pcmBuffer = Buffer.from(base64, 'base64');

    // PCM → WAV（STT 用）
    const sttWavBuffer = pcm16ToWavBuffer(pcmBuffer, { sampleRate: 16000, numChannels: 1 });

    // Azure Speech STT でテキスト化
    const recognizedText = await callAzureSpeechToText(sttWavBuffer);

    // Azure OpenAI Chat で返答を生成
    const replyText = await callAzureOpenAIChat(recognizedText);

    // Azure Speech TTS で WAV(PCM) を生成
    const ttsWavBuffer = await callAzureSpeechTTS(replyText);

    // WAV バイナリをそのまま返す
    res.setHeader('Content-Type', 'audio/wav');
    res.setHeader('Content-Length', ttsWavBuffer.length);

    // 参考用にテキストもヘッダに付けておく（任意）
    res.setHeader('X-Recognized-Text', encodeURIComponent(recognizedText).slice(0, 1024));
    res.setHeader('X-Reply-Text', encodeURIComponent(replyText).slice(0, 1024));

    res.status(200).send(ttsWavBuffer);
  } catch (err) {
    console.error('[pocket-ai] Error:', err);
    res.status(500).json({
      error: 'Internal Server Error',
      message: process.env.NODE_ENV === 'production' ? undefined : String(err?.message || err)
    });
  }
};


