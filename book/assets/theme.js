// Theme toggle
(function() {
  var saved = localStorage.getItem('theme');
  if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.setAttribute('data-theme', 'dark');
  }

  // Close sidebar on click outside (mobile)
  document.addEventListener('click', function(e) {
    if (document.body.classList.contains('sidebar-open')) {
      var sidebar = document.querySelector('.sidebar');
      var toggle = document.querySelector('.sidebar-toggle');
      if (!sidebar.contains(e.target) && !toggle.contains(e.target)) {
        document.body.classList.remove('sidebar-open');
      }
    }
  });

  // Smooth image loading
  document.querySelectorAll('img').forEach(function(img) {
    img.loading = 'lazy';
    img.onerror = function() {
      var dataSrc = this.getAttribute('data-src') || this.getAttribute('data-original');
      if (dataSrc) {
        this.src = dataSrc;
      }
    };
  });
})();
