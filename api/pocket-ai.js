// api/pocket-ai.js
// Vercel Node.js Runtime 用のシンプルなハンドラ :contentReference[oaicite:3]{index=3}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '1mb', // 10秒くらいのPCMなら十分
    },
  },
};

export default async function handler(req, res) {
  try {
    if (req.method !== 'POST') {
      res.status(405).json({ error: 'Use POST' });
      return;
    }

    const { audioBase64 } = req.body || {};
    if (!audioBase64) {
      res.status(400).json({ error: 'audioBase64 is required' });
      return;
    }

    // 1. Base64 -> PCM16バッファ
    const pcmBuffer = Buffer.from(audioBase64, 'base64');

    // 2. PCM16 -> WAV 16kHz mono
    const wavBuffer = pcm16ToWav(pcmBuffer, 16000);

    // 3. Azure Speech で STT
    const text = await speechToText(wavBuffer);

    // デバッグ用：何と言ったかをそのまま返すだけでもOK
    // return res.status(200).json({ text });

    // 4. Azure OpenAI で返答生成
    const reply = await callChatGPT(text || '（聞き取れませんでした）');

    // 5. Azure Speech で TTS
    const audioBuffer = await textToSpeech(reply);

    // 6. MP3 を返す
    res.setHeader('Content-Type', 'audio/mpeg');
    res.send(audioBuffer);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'internal error', detail: String(err) });
  }
}

/**
 * 16bit PCM モノラル -> WAV (RIFF) 変換
 */
function pcm16ToWav(pcmBuffer, sampleRate) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = (sampleRate * numChannels * bitsPerSample) / 8;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const dataSize = pcmBuffer.length;

  const buffer = Buffer.alloc(44 + dataSize);

  let offset = 0;
  buffer.write('RIFF', offset); offset += 4;
  buffer.writeUInt32LE(36 + dataSize, offset); offset += 4;
  buffer.write('WAVE', offset); offset += 4;

  // fmt chunk
  buffer.write('fmt ', offset); offset += 4;
  buffer.writeUInt32LE(16, offset); offset += 4;          // Subchunk1Size
  buffer.writeUInt16LE(1, offset); offset += 2;           // AudioFormat (PCM)
  buffer.writeUInt16LE(numChannels, offset); offset += 2; // NumChannels
  buffer.writeUInt32LE(sampleRate, offset); offset += 4;  // SampleRate
  buffer.writeUInt32LE(byteRate, offset); offset += 4;    // ByteRate
  buffer.writeUInt16LE(blockAlign, offset); offset += 2;  // BlockAlign
  buffer.writeUInt16LE(bitsPerSample, offset); offset += 2; // BitsPerSample

  // data chunk
  buffer.write('data', offset); offset += 4;
  buffer.writeUInt32LE(dataSize, offset); offset += 4;

  pcmBuffer.copy(buffer, offset);

  return buffer;
}

/**
 * Speech STT (短い音声用 REST) :contentReference[oaicite:4]{index=4}
 */
async function speechToText(wavBuffer) {
  const token = await getSpeechToken();
  const region = process.env.AZURE_SPEECH_REGION;

  const url =
    `https://${region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1` +
    `?language=ja-JP`;

  const resp = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000',
      Accept: 'application/json',
    },
    body: wavBuffer
  });

  if (!resp.ok) {
    throw new Error(`STT failed: ${resp.status} ${await resp.text()}`);
  }

  const json = await resp.json();
  return json.DisplayText || '';
}

async function getSpeechToken() {
  const region = process.env.AZURE_SPEECH_REGION;
  const key = process.env.AZURE_SPEECH_KEY;

  const resp = await fetch(
    `https://${region}.api.cognitive.microsoft.com/sts/v1.0/issueToken`,
    {
      method: 'POST',
      headers: {
        'Ocp-Apim-Subscription-Key': key,
        'Content-Type': 'application/x-www-form-urlencoded',
        'Content-Length': '0',
      },
    }
  );

  if (!resp.ok) {
    throw new Error(`getSpeechToken failed: ${resp.status} ${await resp.text()}`);
  }
  return resp.text();
}

/**
 * Azure OpenAI Chat Completions :contentReference[oaicite:5]{index=5}
 */
async function callChatGPT(userText) {
  const endpoint = process.env.AZURE_OPENAI_ENDPOINT;
  const deployment = process.env.AZURE_OPENAI_DEPLOYMENT;
  const apiKey = process.env.AZURE_OPENAI_KEY;
  const apiVersion = '2024-02-15-preview';

  const url = `${endpoint}/openai/deployments/${deployment}/chat/completions?api-version=${apiVersion}`;

  const resp = await fetch(url, {
    method: 'POST',
    headers: {
      'api-key': apiKey,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      messages: [
        {
          role: 'system',
          content: 'あなたは小さなポケットAIアシスタントです。短く、カジュアルに日本語で答えてください。',
        },
        { role: 'user', content: userText },
      ],
      max_tokens: 256,
    }),
  });

  if (!resp.ok) {
    throw new Error(`OpenAI failed: ${resp.status} ${await resp.text()}`);
  }
  const json = await resp.json();
  return json.choices?.[0]?.message?.content?.trim() ?? '';
}

/**
 * Azure Text-to-Speech (REST) :contentReference[oaicite:6]{index=6}
 */
async function textToSpeech(text) {
  const token = await getSpeechToken();
  const region = process.env.AZURE_SPEECH_REGION;

  const url = `https://${region}.tts.speech.microsoft.com/cognitiveservices/v1`;

  // 日本語女性のサンプル: "ja-JP-NanamiNeural" など、ポータルで確認して好きなのに変えてOK
  const ssml = `
    <speak version="1.0" xml:lang="ja-JP">
      <voice xml:lang="ja-JP" name="ja-JP-NanamiNeural">
        ${escapeXml(text)}
      </voice>
    </speak>`.trim();

  const resp = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/ssml+xml',
      'X-Microsoft-OutputFormat': 'audio-16khz-32kbitrate-mono-mp3',
      'User-Agent': 'pocket-ai-assistant',
    },
    body: ssml,
  });

  if (!resp.ok) {
    throw new Error(`TTS failed: ${resp.status} ${await resp.text()}`);
  }

  const arrayBuffer = await resp.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

function escapeXml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
