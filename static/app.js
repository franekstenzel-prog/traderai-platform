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
