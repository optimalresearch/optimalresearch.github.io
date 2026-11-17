
(function(){
 var sp=document.querySelector('.spine'),svg=sp.querySelector('svg'),cur=document.getElementById('cur'),tip=document.getElementById('tip'),
  W=+svg.dataset.w,H=+svg.dataset.h,x0=+svg.dataset.x0,xs=+svg.dataset.xs,band=+svg.dataset.band,
  ticks=JSON.parse(document.getElementById('ticks').textContent);
 function openTo(el){for(var p=el;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true}
 function go(){var id=decodeURIComponent(location.hash.slice(1)),el=id&&document.getElementById(id);if(el){openTo(el);el.scrollIntoView()}}
 addEventListener('hashchange',go);go();
 var pend=false;function spy(){pend=false;var r=sp.getBoundingClientRect(),m=document.querySelector('main').getBoundingClientRect(),
  y=Math.max(r.bottom,0)+30,s=null;for(var i=0;i<12&&!s;i++){var el=document.elementFromPoint(m.left+m.width/2,y+i*12);s=el&&el.closest('[data-t]')}
  if(s){var x=x0+parseFloat(s.dataset.t)*xs;cur.setAttribute('x1',x);cur.setAttribute('x2',x)}}
 addEventListener('scroll',function(){if(!pend){pend=true;requestAnimationFrame(spy)}},{passive:true});spy();
 function near(t){var b=null,d=1e9;for(var i=0;i<ticks.length;i++){var e=Math.abs(ticks[i][0]-t);if(e<d){d=e;b=ticks[i]}}return b}
 function pos(e){var b=svg.getBoundingClientRect();return{x:(e.clientX-b.left)/b.width*W,y:(e.clientY-b.top)/b.height*H,b:b}}
 svg.addEventListener('mousemove',function(e){var p=pos(e),t=e.target.getAttribute&&e.target.getAttribute('data-tip');
  if(!t&&p.y>=band&&ticks.length){var k=near((p.x-x0)/xs);t=k&&k[2]}
  if(!t){tip.hidden=true;return}tip.textContent=t;tip.hidden=false;var s=sp.getBoundingClientRect();
  tip.style.left=Math.max(0,Math.min(e.clientX-s.left+12,s.width-tip.offsetWidth))+'px';tip.style.top=(e.clientY-s.top-34)+'px'});
 svg.addEventListener('mouseleave',function(){tip.hidden=true});
 svg.addEventListener('click',function(e){var p=pos(e);if(p.y>=band&&ticks.length){var k=near((p.x-x0)/xs);if(k)location.hash='s'+k[1]}});
 var btn=document.getElementById('expand'),q=document.getElementById('q'),cnt=document.getElementById('cnt'),rows=[].slice.call(document.querySelectorAll('.ch > .st')),
  chs=[].slice.call(document.querySelectorAll('.ch')),opened=[];
 btn.addEventListener('click',function(){var on=btn.dataset.on!=='1';document.querySelectorAll('details.grp,details.call,details.sub').forEach(function(d){d.open=on});btn.dataset.on=on?'1':'';btn.textContent=on?'Collapse commands':'Expand commands'});
 function hl(s){if(!window.CSS||!CSS.highlights)return;CSS.highlights.delete('hit');if(!s)return;var rs=[];rows.forEach(function(r){if(r.hidden||rs.length>3000)return;
  var w=document.createTreeWalker(r,NodeFilter.SHOW_TEXT),n;while((n=w.nextNode())){var v=n.nodeValue.toLowerCase(),i=v.indexOf(s);while(i>=0&&rs.length<=3000){var g=new Range();g.setStart(n,i);g.setEnd(n,i+s.length);rs.push(g);i=v.indexOf(s,i+s.length)}}});
  var h=new Highlight();rs.forEach(function(g){h.add(g)});CSS.highlights.set('hit',h)}
 var tm;q.addEventListener('input',function(){clearTimeout(tm);tm=setTimeout(function(){var s=q.value.trim().toLowerCase(),n=0;
  opened.forEach(function(d){d.open=false});opened=[];
  rows.forEach(function(r){var m=!s||r.textContent.toLowerCase().indexOf(s)>=0;r.hidden=!m;if(m)n++;
   if(m&&s)r.querySelectorAll('details').forEach(function(d){if(!d.open&&d.textContent.toLowerCase().indexOf(s)>=0){d.open=true;opened.push(d)}})});
  chs.forEach(function(c){c.hidden=!!s&&!c.querySelector(':scope > .st:not([hidden])')});
  cnt.textContent=s?n+' of '+rows.length+' match':'';hl(s);spy()},120)});
})();
