// Vercel Serverless Function
// AtomS3R + Echo Base から送られてきた Base64 PCM(16kHz/16bit/mono) を
// 1. WAV に変換して Azure Speech STT で文字起こし
// 2. Azure OpenAI (Azure AI Foundry) で TODO かどうかを判定・構造化
// 3. TODO のときだけ Notion Webhook に送信
// 4. デバイスには recognizedText と todoTitle などを JSON で返す

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
  const recognizedText = data.DisplayText || data.Text || '';
  return recognizedText.trim();
}

async function callAzureOpenAITodoAnalyzer(text) {
  const openaiKey = process.env.AZURE_OPENAI_KEY;
  const openaiEndpoint = process.env.AZURE_OPENAI_ENDPOINT;
  const openaiDeployment = process.env.AZURE_OPENAI_DEPLOYMENT;
  const apiVersion = process.env.AZURE_OPENAI_API_VERSION || '2024-02-15-preview';

  if (!openaiKey || !openaiEndpoint || !openaiDeployment) {
    throw new Error('AZURE_OPENAI_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT が設定されていません。');
  }

  const endpoint = openaiEndpoint.replace(/\/$/, '');
  const url = `${endpoint}/openai/deployments/${openaiDeployment}/chat/completions?api-version=${apiVersion}`;

  const systemPrompt =
    'あなたは日本語の会話からTODOを抽出するアシスタントです。' +
    '入力は1フレーズ〜数フレーズの口語文です。' +
    '「〜を買いたい」「〜に行く」「〜をやる」など、明らかにユーザーの行動を示す内容だけをTODOとして扱ってください。' +
    '雑談や感想だけの場合はTODOにはしません。' +
    '出力は必ず次のJSONだけにしてください。' +
    '{"is_todo": boolean, "title": string | null, "when": string | null, "notes": string | null} ' +
    'title にはTODOとして短く要約した行動（例: "玉ねぎを買う"）を入れ、when には「今日中」「明日」「来週末」などを入れてください。' +
    'notes には元の発話の内容や補足を簡潔に入れてください。';

  const body = {
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: text }
    ],
    max_tokens: 256,
    temperature: 0.2
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
    const t = await res.text().catch(() => '');
    throw new Error(`Azure OpenAI TODO 解析でエラーが発生しました: ${res.status} ${res.statusText} ${t}`);
  }

  const data = await res.json();
  let content =
    data.choices?.[0]?.message?.content?.trim() ||
    data.choices?.[0]?.messages?.[0]?.content?.trim() ||
    '';

  if (!content) {
    throw new Error('Azure OpenAI からのTODO解析結果が空でした。');
  }

  // ```json ... ``` 形式で返ってきた場合に備えてトリミング
  if (content.startsWith('```')) {
    const firstNewline = content.indexOf('\n');
    const lastFence = content.lastIndexOf('```');
    if (firstNewline >= 0 && lastFence > firstNewline) {
      content = content.substring(firstNewline + 1, lastFence).trim();
    }
  }

  let parsed;
  try {
    parsed = JSON.parse(content);
  } catch (e) {
    // 失敗した場合はTODOなし扱いにフォールバック
    return {
      is_todo: false,
      title: null,
      when: null,
      notes: null
    };
  }

  return {
    is_todo: !!parsed.is_todo,
    title: parsed.title ?? null,
    when: parsed.when ?? null,
    notes: parsed.notes ?? null
  };
}

async function postTodoToNotion({ title, when, notes, sourceText, sessionId }) {
  const notionKey = process.env.NOTION_API_KEY;
  const databaseId = process.env.NOTION_DATABASE_ID;

  if (!notionKey || !databaseId) {
    console.warn('[todo-from-speech] NOTION_API_KEY / NOTION_DATABASE_ID が設定されていないため、Notion には保存しません。');
    return false;
  }

  const url = 'https://api.notion.com/v1/pages';

  const properties = {
    Title: {
      title: [
        {
          text: {
            content: title || (sourceText || '').slice(0, 50) || 'Speech TODO'
          }
        }
      ]
    }
  };

  if (when) {
    properties.When = {
      rich_text: [
        {
          text: {
            content: when
          }
        }
      ]
    };
  }

  const children = [];
  if (notes) {
    children.push({
      object: 'block',
      type: 'paragraph',
      paragraph: {
        rich_text: [
          {
            type: 'text',
            text: { content: `Notes: ${notes}` }
          }
        ]
      }
    });
  }
  if (sourceText) {
    children.push({
      object: 'block',
      type: 'paragraph',
      paragraph: {
        rich_text: [
          {
            type: 'text',
            text: { content: `Source: ${sourceText}` }
          }
        ]
      }
    });
  }
  if (sessionId) {
    children.push({
      object: 'block',
      type: 'paragraph',
      paragraph: {
        rich_text: [
          {
            type: 'text',
            text: { content: `Session: ${sessionId}` }
          }
        ]
      }
    });
  }

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${notionKey}`,
        'Content-Type': 'application/json',
        'Notion-Version': '2022-06-28'
      },
      body: JSON.stringify({
        parent: { database_id: databaseId },
        properties,
        children
      })
    });

    if (!res.ok) {
      const text = await res.text().catch(() => '');
      console.error('[todo-from-speech] Notion API error:', res.status, res.statusText, text);
      return false;
    }

    return true;
  } catch (e) {
    console.error('[todo-from-speech] Notion API request failed:', e);
    return false;
  }
}

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method Not Allowed. Use POST.' });
    return;
  }

  try {
    const body = typeof req.body === 'string' ? JSON.parse(req.body || '{}') : (req.body || {});

    const mode = body.mode || 'utterance';
    if (mode !== 'utterance') {
      res.status(400).json({ error: 'Unsupported mode', mode });
      return;
    }

    const base64 = body.audioBase64 || body.audio || '';
    if (!base64 || typeof base64 !== 'string') {
      res.status(400).json({ error: 'audioBase64 (Base64 PCM) が指定されていません。' });
      return;
    }

    const sessionId = body.sessionId || null;

    // Base64 → PCM Buffer
    const pcmBuffer = Buffer.from(base64, 'base64');

    // PCM → WAV
    const wavBuffer = pcm16ToWavBuffer(pcmBuffer, { sampleRate: 16000, numChannels: 1 });

    // Azure Speech STT でテキスト化
    const recognizedText = await callAzureSpeechToText(wavBuffer);

    let todo = {
      is_todo: false,
      title: null,
      when: null,
      notes: null
    };
    let notionPosted = false;

    if (recognizedText) {
      todo = await callAzureOpenAITodoAnalyzer(recognizedText);

      if (todo.is_todo) {
        notionPosted = await postTodoToNotion({
          title: todo.title,
          when: todo.when,
          notes: todo.notes,
          sourceText: recognizedText,
          sessionId
        });
      }
    }

    res.status(200).json({
      recognizedText,
      todoTitle: todo.title,
      todoWhen: todo.when,
      todoNotes: todo.notes,
      notionPosted
    });
  } catch (err) {
    console.error('[todo-from-speech] Error:', err);
    res.status(500).json({
      error: 'Internal Server Error',
      message: process.env.NODE_ENV === 'production' ? undefined : String(err?.message || err)
    });
  }
};

