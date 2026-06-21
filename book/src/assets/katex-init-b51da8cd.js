document.addEventListener("DOMContentLoaded", function() {
  if (typeof renderMathInElement === 'undefined' || typeof katex === 'undefined') {
    console.warn('KaTeX not loaded, skipping math rendering');
    return;
  }
  renderMathInElement(document.body, {
    delimiters: [
      {left: '$$', right: '$$', display: true},
      {left: '$', right: '$', display: false},
      {left: '\\(', right: '\\)', display: false},
      {left: '\\[', right: '\\]', display: true}
    ],
    throwOnError: false,
    trust: true,
    strict: false
  });
});
