#!/usr/bin/env node
const jsBeautify = require('js-beautify');
const fs = require('fs');
const path = require('path');

const BOOK_DIR = path.join(__dirname, '..', 'book');

const HTML_OPTIONS = {
  indent_size: 2,
  indent_char: ' ',
  max_preserve_newlines: 1,
  preserve_newlines: true,
  unformatted: ['code', 'pre', 'span'],
  content_unformatted: ['pre', 'code'],
  indent_inner_html: true,
  wrap_line_length: 0,
  end_with_newline: true,
};

function findHtmlFiles(dir) {
  const files = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...findHtmlFiles(full));
    } else if (entry.name.endsWith('.html') || entry.name.endsWith('.htm')) {
      files.push(full);
    }
  }
  return files;
}

function formatFile(filepath) {
  const raw = fs.readFileSync(filepath, 'utf-8');
  const formatted = jsBeautify.html(raw, HTML_OPTIONS);
  if (formatted !== raw) {
    fs.writeFileSync(filepath, formatted, 'utf-8');
    return true;
  }
  return false;
}

const files = findHtmlFiles(BOOK_DIR);
let changed = 0;
for (const f of files) {
  if (formatFile(f)) {
    changed++;
    console.log(`  formatted: ${path.relative(BOOK_DIR, f)}`);
  }
}
console.log(`\nDone: ${changed}/${files.length} files formatted`);
