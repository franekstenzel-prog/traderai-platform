// Minimal slider: 3 steps
(function(){
  const root = document.querySelector('[data-slider]');
  if(!root) return;

  const slides = Array.from(root.querySelectorAll('.slide'));
  const prev = root.querySelector('[data-prev]');
  const next = root.querySelector('[data-next]');
  let i = 0;

  function render(){
    slides.forEach((s, idx) => s.classList.toggle('is-active', idx === i));
  }
  function step(dir){
    i = (i + dir + slides.length) % slides.length;
    render();
  }

  prev && prev.addEventListener('click', () => step(-1));
  next && next.addEventListener('click', () => step(1));

  render();
})();

// App shell: mobile sidebar toggle
(function(){
  const btn = document.querySelector('[data-sidebar-toggle]');
  const shell = document.querySelector('[data-app-shell]');
  const overlay = document.querySelector('[data-sidebar-overlay]');
  if(!btn || !shell) return;

  function setOpen(isOpen){
    shell.classList.toggle('is-nav-open', !!isOpen);
    if(overlay) overlay.classList.toggle('is-active', !!isOpen);
    document.body.classList.toggle('no-scroll', !!isOpen);
  }

  btn.addEventListener('click', () => {
    setOpen(!shell.classList.contains('is-nav-open'));
  });
  overlay && overlay.addEventListener('click', () => setOpen(false));

  // Close on Esc
  document.addEventListener('keydown', (e) => {
    if(e.key === 'Escape') setOpen(false);
  });
})();
