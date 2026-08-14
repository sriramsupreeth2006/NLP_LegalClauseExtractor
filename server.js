const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');

const app = express();
const port = process.env.PORT || 3000;
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 25 * 1024 * 1024 } });

app.use(cors({ origin: true }));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(express.static(__dirname));

app.get('/', (req, res) => {
  res.redirect('/clause-extractor.html');
});

app.post('/api/extract', upload.single('file'), async (req, res) => {
  try {
    const text = req.file ? await extractTextFromFile(req.file) : String(req.body.text || '');
    const normalizedText = normalizeText(text);

    if (!normalizedText.trim()) {
      return res.status(400).json({ error: 'Provide contract text or upload a supported file.' });
    }

    const clauses = extractClauses(normalizedText);
    res.json({ text: normalizedText, clauses });
  } catch (error) {
    res.status(500).json({ error: error.message || 'Extraction failed.' });
  }
});

app.listen(port, () => {
  console.log(`Legal clause extractor backend running at http://localhost:${port}`);
});

async function extractTextFromFile(file) {
  const extension = path.extname(file.originalname || '').toLowerCase();

  if (extension === '.pdf') {
    const result = await pdfParse(file.buffer);
    return result.text || '';
  }

  if (extension === '.docx') {
    const result = await mammoth.extractRawText({ buffer: file.buffer });
    return result.value || '';
  }

  if (extension === '.html' || extension === '.htm') {
    return file.buffer.toString('utf8').replace(/<script[\s\S]*?<\/script>/gi, ' ').replace(/<style[\s\S]*?<\/style>/gi, ' ').replace(/<[^>]+>/g, ' ');
  }

  return file.buffer.toString('utf8');
}

function normalizeText(text) {
  return String(text || '').replace(/\r\n/g, '\n').replace(/\u0000/g, '').trim();
}

function extractClauses(text) {
  const segments = splitSegments(text)
    .map(segment => segment.trim())
    .filter(Boolean);

  const clauses = [];

  for (const segment of segments) {
    const category = classifySegment(segment);
    if (category === null) {
      continue;
    }

    clauses.push({
      category,
      title: inferTitle(segment, category),
      summary: summarizeSegment(segment, category),
      section: inferSection(segment, clauses.length + 1),
      excerpt: segment,
    });
  }

  return clauses;
}

function splitSegments(text) {
  const normalized = normalizeText(text);
  const paragraphs = normalized.split(/\n\s*\n+/).map(part => part.trim()).filter(Boolean);

  if (paragraphs.length > 1) {
    return paragraphs;
  }

  const lineBased = normalized
    .split(/\n(?=\s*(?:\d+(?:\.\d+)*[.)]?|Section\s+\d+(?:\.\d+)*|[A-Z][A-Za-z0-9 ,/&()\-]{4,}:)\s+)/i)
    .map(part => part.trim())
    .filter(Boolean);

  if (lineBased.length > 1) {
    return lineBased;
  }

  return normalized
    .split(/(?<=[.?!])\s+(?=(?:\d+(?:\.\d+)*[.)]?|Section\s+\d+(?:\.\d+)*|[A-Z][A-Za-z0-9 ,/&()\-]{4,}:)\s+)/i)
    .map(part => part.trim())
    .filter(Boolean)
    .length > 1
    ? normalized.split(/(?<=[.?!])\s+(?=(?:\d+(?:\.\d+)*[.)]?|Section\s+\d+(?:\.\d+)*|[A-Z][A-Za-z0-9 ,/&()\-]{4,}:)\s+)/i).map(part => part.trim()).filter(Boolean)
    : normalized.split(/\n+/).map(part => part.trim()).filter(Boolean);
}

function classifySegment(segment) {
  const lower = segment.toLowerCase();

  if (/liability cap|aggregate liability|limitation of liability/.test(lower)) {
    return 'other';
  }

  if (/indemn|confidential|warranty|insurance|governing law|audit|compliance|privacy/.test(lower)) {
    return 'other';
  }

  if (/early termination fee|termination fee|late fee|liquidated damages|penalt|suspend|non-payment|default/.test(lower)) {
    return 'penalties';
  }

  if (/terminate|termination|renewal|expiry|expire|notice|breach|cure|convenience/.test(lower)) {
    return 'termination';
  }

  if (/payment|invoice|billing|remit|due|monthly fee|fees?/.test(lower)) {
    return 'payment';
  }

  const scores = {
    termination: score(lower, ['terminate', 'termination', 'renewal', 'expiry', 'expire', 'notice', 'breach', 'cure', 'convenience', 'assistance in transition']),
    payment: score(lower, ['payment', 'pay', 'invoice', 'fee', 'fees', 'billing', 'due', 'net ', 'monthly', 'remit', 'amount owed']),
    penalties: score(lower, ['penalty', 'penalties', 'late fee', 'liquidated damages', 'interest', 'late payment', 'suspend', 'default', 'withhold']),
    other: score(lower, ['liability', 'indemn', 'confidential', 'warranty', 'insurance', 'compliance', 'intellectual property', 'audit', 'governing law']),
  };

  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  if (!entries.length || entries[0][1] === 0) {
    return null;
  }

  return entries[0][0];
}

function score(text, keywords) {
  return keywords.reduce((total, keyword) => {
    const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const matches = text.match(new RegExp(escaped, 'gi'));
    return total + (matches ? matches.length : 0);
  }, 0);
}

function inferTitle(segment, category) {
  const cleaned = segment.replace(/\s+/g, ' ').trim();
  const sectionMatch = cleaned.match(/^\s*(?:Section\s+\d+(?:\.\d+)*|\d+(?:\.\d+)*[.)]?|[IVXLC]+[.)]?)\s*([^:.]{4,80}?)(?:[.:]|\s{2,}|\s+-\s+|$)/i);
  const rawTitle = sectionMatch ? sectionMatch[1] : cleaned.split(/[.?!:;]/)[0];
  const words = rawTitle.replace(/^[-\d\s.()]+/, '').split(/\s+/).filter(Boolean).slice(0, 5);

  if (words.length > 0) {
    return words.join(' ');
  }

  const fallback = {
    termination: 'Termination clause',
    payment: 'Payment terms',
    penalties: 'Penalty clause',
    other: 'Risk clause',
  };

  return fallback[category] || 'Clause';
}

function summarizeSegment(segment, category) {
  const lower = segment.toLowerCase();
  const noticeDays = findQuantity(segment, /(?:upon|with|within|prior written notice|notice of|after|for)\s+((?:\d+(?:\.\d+)?)|(?:[a-z]+(?:[-\s][a-z]+)*))(?:\s*\(\d+(?:\.\d+)?\))?\s*(?:days?|business days?)/i);
  const cureDays = findQuantity(segment, /cure[^.]{0,80}?(?:within\s+)?((?:\d+(?:\.\d+)?)|(?:[a-z]+(?:[-\s][a-z]+)*))(?:\s*\(\d+(?:\.\d+)?\))?\s*(?:days?|business days?)/i);
  const dueDays = findQuantity(segment, /(?:due[^.]{0,80}?within|within)\s+((?:\d+(?:\.\d+)?)|(?:[a-z]+(?:[-\s][a-z]+)*))(?:\s*\(\d+(?:\.\d+)?\))?\s*(?:days?|business days?)/i);
  const feePercent = findFirstNumber(segment, /(\d+(?:\.\d+)?)\s*%/i);
  const monthlyFee = findFirstMoney(segment, /\$\s*([\d,]+(?:\.\d+)?)/i);
  const interestRate = findFirstNumber(segment, /(\d+(?:\.\d+)?)\s*%\s*(?:per\s*month|monthly|per\s*annum|annually)/i);

  if (category === 'termination') {
    if (/for convenience/.test(lower) && noticeDays) {
      return `Either party may terminate for convenience with ${noticeDays} days' notice.`;
    }

    if (/immediate|immediately/.test(lower) && cureDays) {
      return `A party may terminate immediately if the other side does not cure a breach within ${cureDays} days.`;
    }

    if (/breach/.test(lower) && cureDays) {
      return `The agreement can end if a material breach is not cured within ${cureDays} days.`;
    }

    if (noticeDays) {
      return `The agreement can be terminated with ${noticeDays} days' notice.`;
    }

    return 'This clause explains when the agreement can be ended.';
  }

  if (category === 'payment') {
    const paymentParts = [];

    if (monthlyFee) {
      paymentParts.push(`Client must pay ${monthlyFee} on the stated billing schedule`);
    }

    if (dueDays) {
      paymentParts.push(`payment is due within ${dueDays} days of invoice`);
    }

    if (interestRate) {
      paymentParts.push(`late balances accrue ${interestRate}% interest`);
    }

    if (paymentParts.length) {
      return paymentParts.join(', ') + '.';
    }

    return 'This clause sets the payment schedule and invoice timing.';
  }

  if (category === 'penalties') {
    if (/early termination fee|termination fee/.test(lower)) {
      return 'Client must pay an early termination fee if the agreement ends before the full term is complete.';
    }

    const penaltyParts = [];

    if (feePercent) {
      penaltyParts.push(`late charges include a ${feePercent}% fee`);
    }

    if (/suspend/.test(lower)) {
      penaltyParts.push('services may be suspended for non-payment');
    }

    if (interestRate) {
      penaltyParts.push(`interest accrues at ${interestRate}%`);
    }

    if (penaltyParts.length) {
      return penaltyParts.join(', ') + '.';
    }

    return 'This clause imposes a penalty or late-payment consequence.';
  }

  if (/liability cap|aggregate liability|limitation of liability/.test(lower)) {
    return 'This clause caps the parties liability to a specified amount.';
  }

  if (/indemn/.test(lower)) {
    return 'This clause requires one party to protect the other from certain claims or losses.';
  }

  if (/confidential/.test(lower)) {
    return 'This clause restricts how confidential information may be used or shared.';
  }

  return 'This clause creates an important contractual obligation or risk allocation.';
}

function inferSection(segment, fallbackIndex) {
  const text = segment.trim();
  const explicit = text.match(/^\s*(Section\s+\d+(?:\.\d+)*|\d+(?:\.\d+)*[.)]?|§\s*\d+(?:\.\d+)*)\b/i);

  if (explicit) {
    const label = explicit[1].replace(/\s+/g, ' ').trim();
    return label.startsWith('Section') || label.startsWith('§') ? label : `Section ${label.replace(/[.)]$/, '')}`;
  }

  return `Section ${fallbackIndex}`;
}

function findFirstNumber(text, regex) {
  const match = text.match(regex);
  return match ? match[1] : '';
}

function findFirstMoney(text, regex) {
  const match = text.match(regex);
  return match ? `$${match[1]}` : '';
}

function findQuantity(text, regex) {
  const match = text.match(regex);
  if (!match) {
    return '';
  }

  return normalizeQuantity(match[1]);
}

function normalizeQuantity(fragment) {
  const digits = String(fragment).match(/\d+(?:\.\d+)?/);
  if (digits) {
    return digits[0];
  }

  const value = wordsToNumber(String(fragment));
  return value ? String(value) : '';
}

function wordsToNumber(fragment) {
  const tokens = String(fragment)
    .toLowerCase()
    .replace(/[^a-z\s-]/g, ' ')
    .split(/[\s-]+/)
    .filter(Boolean);

  const units = {
    zero: 0,
    one: 1,
    two: 2,
    three: 3,
    four: 4,
    five: 5,
    six: 6,
    seven: 7,
    eight: 8,
    nine: 9,
    ten: 10,
    eleven: 11,
    twelve: 12,
    thirteen: 13,
    fourteen: 14,
    fifteen: 15,
    sixteen: 16,
    seventeen: 17,
    eighteen: 18,
    nineteen: 19,
  };

  const tens = {
    twenty: 20,
    thirty: 30,
    forty: 40,
    fifty: 50,
    sixty: 60,
    seventy: 70,
    eighty: 80,
    ninety: 90,
  };

  let total = 0;
  let current = 0;

  for (const token of tokens) {
    if (token === 'and') {
      continue;
    }

    if (units[token] !== undefined) {
      current += units[token];
      continue;
    }

    if (tens[token] !== undefined) {
      current += tens[token];
      continue;
    }

    if (token === 'hundred') {
      current = (current || 1) * 100;
      continue;
    }

    if (current) {
      total += current;
      current = 0;
    }
  }

  total += current;
  return total;
}