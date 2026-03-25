// === Scroll Reveal ===
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
    }
  });
}, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

// === Nav scroll effect ===
const nav = document.getElementById('nav');
if (nav) {
  window.addEventListener('scroll', () => {
    nav.classList.toggle('scrolled', window.scrollY > 50);
  }, { passive: true });
}

// === Mobile nav toggle ===
const toggle = document.getElementById('nav-toggle');
const links = document.getElementById('nav-links');
if (toggle && links) {
  toggle.addEventListener('click', () => links.classList.toggle('open'));
  // Close on link click
  links.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', () => links.classList.remove('open'));
  });
}

// === Smooth scroll for anchor links ===
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function(e) {
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});
