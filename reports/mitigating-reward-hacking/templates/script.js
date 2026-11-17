(function(){
 function open(){var h=decodeURIComponent(location.hash.slice(1)),el=h&&document.getElementById(h);
  if(el&&el.tagName==='DETAILS')el.open=true}
 addEventListener('hashchange',open);open();
 document.querySelectorAll('.chart,.spark').forEach(function(f){var tip=f.querySelector('.tip');
  function on(e){var t=e.target.getAttribute&&e.target.getAttribute('data-tip');if(!t)return;tip.textContent=t;tip.hidden=false;
   var b=f.getBoundingClientRect(),r=e.target.getBoundingClientRect();
   tip.style.left=Math.max(0,Math.min(r.left-b.left+12,b.width-tip.offsetWidth))+'px';tip.style.top=(r.top-b.top-34)+'px';}
  function off(){tip.hidden=true}
  f.addEventListener('mouseover',on);f.addEventListener('focusin',on);f.addEventListener('mouseout',off);f.addEventListener('focusout',off);});
})();
