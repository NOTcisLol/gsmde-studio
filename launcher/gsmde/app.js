/* GSMDE Studio — front de producao. Reusa a API pywebview do launcher
   (gsmde_start / gsmde_status / gsmde_route / gsmde_centros / gsmde_yolos /
    gsmde_pick_meta / gsmde_stop). */
"use strict";
const NL1 = String.fromCharCode(10), NL2 = NL1 + NL1;
const $ = (id) => document.getElementById(id);
const api = () => (window.pywebview && window.pywebview.api) || null;
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));
let generating = false, pollTimer = null, pvSeen = -1, etapasSeen = 0;
let rodada = 0, pararLoop = false;
let CENTROS = {};            // nome -> {tags, global, cluster}
let CAMADAS = [{g:1.5, dn:0.45, st:30, on:true}];
let DETS = [];               // [{model, prompt, dn}]
let YOLOS = [];              // [{id, nome}]

/* ---------- balao de ajuda (canto sup. direito): FIXA ao sair do mouse ---------- */
let hintOn = true;
function setHint(on){
  hintOn = !!on;
  $("help").style.display = hintOn ? "" : "none";
  document.body.classList.toggle("sem-hint", !hintOn);   // a barra ocupa o espaco
  $("hintBtn").classList.toggle("on", hintOn);
  $("hintBtn").title = hintOn ? "esconder a ajuda do canto" : "mostrar a ajuda do canto";
}
let CAMINHOS_TXT = "";

/* Alguns verbetes mostram estado ao vivo (ex.: de onde os pesos vem). `dyn`
   nomeia uma funcao global cujo retorno e' anexado ao corpo do balao. */
window.HELP_DYN = {
  caminhos: () => CAMINHOS_TXT || "(medindo… gere uma vez para preencher)",
  ramBuffer: () => RAM_DIAG
};

function showHelp(id, forcar){
  // `forcar` existe para o alerta de memoria: um usuario novo provavelmente
  // desligou o hint e nao faz ideia de ONDE arrumar o problema. O balao e' o
  // unico lugar que explica o porque. Silenciar o som e' opcao; ficar sem a
  // explicacao, nao.
  if(!hintOn && !forcar) return;
  if(forcar){
    $("help").style.display = "";
    document.body.classList.remove("sem-hint");
  }
  const h = (window.HELP && window.HELP[id]) || null;
  if(!h) return;
  $("h-title").textContent = h.t || "";
  const extra = (window.HELP_DYN && window.HELP_DYN[id]) ? window.HELP_DYN[id]() : "";
  $("h-body").textContent = (h.b || "") + (extra ? NL2 + extra : "");
  const line = (el, val) => {
    const box = $(el); if(val){ box.style.display=""; box.querySelector(".v").textContent = val; }
    else box.style.display="none";
  };
  line("h-img", h.img && h.img !== "—" ? h.img : "");
  line("h-hw",  h.hw);
  const hint = $("h-hint");
  if(h.hint){ hint.style.display=""; hint.textContent = "💡 " + h.hint; } else hint.style.display="none";
  const help = $("help"); help.classList.add("lit");
  clearTimeout(help._t); help._t = setTimeout(()=>help.classList.remove("lit"), 400);
}
document.addEventListener("mouseover", (e)=>{
  const el = e.target.closest("[data-help]");
  if(el) showHelp(el.getAttribute("data-help"));
});
document.addEventListener("focusin", (e)=>{
  const el = e.target.closest && e.target.closest("[data-help]");
  if(el) showHelp(el.getAttribute("data-help"));
});

/* ---------- sliders <-> rotulos ---------- */
[["escala",2],["gw",2],["backbone",2]].forEach(([id,dec])=>{
  const el = $(id); if(!el) return;
  const upd = ()=> $(id+"_v").textContent = (+el.value).toFixed(dec);
  el.addEventListener("input", upd); upd();
});

/* ---------- helpers de UI ---------- */
function setBusy(b, stage){
  generating = b;
  $("genBtn").disabled = b;
  // O ⏹ NAO segue o estado da UI: o worker e' outro processo e pode estar vivo
  // (ou travado) enquanto a UI acha que nao ha nada. Era justo nessa hora que o
  // botao ficava cinza — e a unica saida virava fechar o app. Parar sem job em
  // andamento e' inofensivo: o gsmde_stop so responde "nao havia nada".
  if(stage!==undefined) $("stage").textContent = stage;
}
function showImg(src){
  $("viewEmpty").style.display="none"; $("logBox").style.display="none";
  $("viewImg").src=src; $("viewImg").style.display="block";
}
function showLog(txt){
  $("logBox").textContent = txt || "(sem log)";
  $("logBox").style.display = "block"; $("viewEmpty").style.display="none";
}
function toast(msg){ $("stage").textContent = msg; }
/* aviso PERSISTENTE: coisas que mudam o resultado (ex.: gerou sem especialista
   nenhum) nao podem viver so na barra de status, que a proxima etapa sobrescreve
   em segundos. Fica visivel ate o usuario fechar. */
function avisa(msg){
  const cx = $("avisos");
  // avisa("") e' usado como "deu certo, limpe os alertas anteriores". Sem esta
  // guarda ele criava uma faixa amarela VAZIA — um alerta sem texto, que so'
  // assusta.
  if(!msg){
    cx.querySelectorAll(".aviso").forEach(e=> e.remove());
    return;
  }
  const el = document.createElement("div");
  el.className = "aviso";
  el.innerHTML = `<span>⚠️ ${esc(msg)}</span><button class="mini-btn">✕</button>`;
  el.querySelector("button").addEventListener("click", ()=> el.remove());
  cx.appendChild(el);
  setTimeout(()=>{ if(el.isConnected) el.remove(); }, 20000);
}

/* ================= resolucao livre =================
   Largura e altura independentes: qualquer proporcao (16:9, 2:7, 1:1...) sem lista
   fechada. O latente e' px/8, entao o worker arredonda p/ multiplo de 8. */
function baseW(){ return Math.max(256, +$("largura").value || 768); }
function baseH(){ return Math.max(256, +$("altura").value || 768); }
function proporcao(w, h){
  const mdc = (a,b)=> b ? mdc(b, a%b) : a;
  const g = mdc(w, h) || 1;
  let rw = w/g, rh = h/g;
  if(rw > 40 || rh > 40){            // 1234:769 nao diz nada — mostra decimal
    return (w/h).toFixed(2) + ":1";
  }
  return `${rw}:${rh}`;
}
function atualizaProp(){
  const w = baseW(), h = baseH();
  const w8 = Math.round(w/8)*8, h8 = Math.round(h/8)*8;
  $("propInfo").textContent = `${w8}×${h8} · ${proporcao(w8,h8)} · `
    + `${(w8*h8/1e6).toFixed(2)} MP` + ((w8!==w || h8!==h) ? "  (ajustado p/ múltiplo de 8)" : "");
  pintaCamadas();                     // o px alvo de cada camada vem do ganho x largura
  if($("custo")) atualizaCusto();
}

/* ================= camadas de hires =================
   Ganho (x vezes), nao resolucao absoluta: com largura/altura livres, "1152px"
   nao diz nada sozinho — o que o usuario pensa e' "1,5x maior". O formato que vai
   p/ o motor continua px absoluto (largura alvo), calculado aqui. */
function pxDaCamada(c){ return Math.max(256, Math.round(baseW() * (+c.g || 1.5) / 8) * 8); }
function pintaCamadas(){
  const h = baseH() / baseW();
  $("camadasBox").innerHTML = CAMADAS.map((c,i)=>{
    const px = pxDaCamada(c);
    return `
    <div class="camada">
      <label class="chk"><input type="checkbox" class="cmOn" data-i="${i}" ${c.on?"checked":""}>
        <b>Camada ${i+1}</b>
        <span class="badge dim">${px}×${Math.round(px*h/8)*8}</span></label>
      <div class="row">
        <div class="field"><div class="lbl">Ganho ×</div>
          <input class="cmG" data-i="${i}" type="number" step="0.05" min="1" max="4" value="${c.g}"></div>
        <div class="field"><div class="lbl">Denoise</div>
          <input class="cmDn" data-i="${i}" type="number" step="0.01" min="0.1" max="0.8" value="${c.dn}"></div>
        <div class="field"><div class="lbl">Passos</div>
          <input class="cmSt" data-i="${i}" type="number" min="6" max="60" value="${c.st}"></div>
      </div>
    </div>`;
  }).join("");
  const bind = (cls, campo)=> $("camadasBox").querySelectorAll("."+cls).forEach(el=>{
    el.addEventListener("input", ()=>{
      CAMADAS[+el.dataset.i][campo] = (campo === "on") ? el.checked : +el.value;
      if(campo === "g"){               // atualiza so o rotulo: repintar roubaria o foco
        const b = el.closest(".camada").querySelector(".badge");
        const px = pxDaCamada(CAMADAS[+el.dataset.i]);
        if(b) b.textContent = `${px}×${Math.round(px*baseH()/baseW()/8)*8}`;
      }
    });
  });
  bind("cmDn","dn"); bind("cmSt","st"); bind("cmG","g"); bind("cmOn","on");
}
function strCamadas(){
  return $("hires").checked
    ? CAMADAS.filter(c=>c.on).map(c=>`${pxDaCamada(c)}:${c.dn}:${c.st}`).join(",") : "";
}

/* ================= detailer: um prompt POR ALVO ================= */
function pintaDets(){
  const nome = (id)=> (YOLOS.find(y=>y.id===id) || {}).nome || id;
  $("detList").innerHTML = DETS.map((d,i)=>`
    <div class="det" draggable="true" data-i="${i}">
      <div class="lbl det-h">
        <span class="det-ord" title="ordem de execução">${i+1}</span>
        <span class="det-pega" title="arraste para reordenar">⠿</span>
        <b>${esc(nome(d.model))}</b>
        <button class="mini-btn detDel" data-i="${i}" title="remover">✕</button></div>
      <textarea class="detP" data-i="${i}" rows="2"
        placeholder="prompt só deste alvo — vazio = usa o positivo">${esc(d.prompt||"")}</textarea>
      <div class="row">
        <div class="field"><div class="lbl">Denoise</div>
          <input class="detDn" data-i="${i}" type="number" step="0.05" min="0.1" max="0.7" value="${d.dn}"></div>
        <div class="field"><div class="lbl">Passos</div>
          <input class="detSt" data-i="${i}" type="number" min="6" max="40" value="${d.st}"></div>
      </div>
    </div>`).join("");
  $("detList").querySelectorAll(".detP").forEach(el=> el.addEventListener("input", ()=>{
    DETS[+el.dataset.i].prompt = el.value; }));
  $("detList").querySelectorAll(".detDn").forEach(el=> el.addEventListener("input", ()=>{
    DETS[+el.dataset.i].dn = +el.value; }));
  $("detList").querySelectorAll(".detSt").forEach(el=> el.addEventListener("input", ()=>{
    DETS[+el.dataset.i].st = +el.value; }));
  $("detList").querySelectorAll(".detDel").forEach(el=> el.addEventListener("click", ()=>{
    DETS.splice(+el.dataset.i,1); pintaDets(); }));

  // ---- reordenar arrastando ----
  // A ORDEM E' SEMANTICA, nao estetica: o detailer roda um alvo por vez e cada
  // um reescreve a sua regiao. Se 'olhos' roda antes de 'cabeca', a mascara da
  // cabeca passa por cima do que os olhos acabaram de refinar — o trabalho do
  // primeiro e' perdido e o tempo, gasto a toa. Por isso o de cima roda antes:
  // o generico primeiro, o detalhe fino depois.
  let arrastando = null;
  $("detList").querySelectorAll(".det").forEach(el=>{
    el.addEventListener("dragstart", e=>{
      arrastando = +el.dataset.i;
      el.classList.add("arrastando");
      e.dataTransfer.effectAllowed = "move";
      try{ e.dataTransfer.setData("text/plain", String(arrastando)); }catch(_){}
    });
    el.addEventListener("dragend", ()=>{
      el.classList.remove("arrastando");
      $("detList").querySelectorAll(".det").forEach(x=> x.classList.remove("alvo"));
    });
    el.addEventListener("dragover", e=>{
      e.preventDefault();
      if(arrastando !== null && +el.dataset.i !== arrastando) el.classList.add("alvo");
    });
    el.addEventListener("dragleave", ()=> el.classList.remove("alvo"));
    el.addEventListener("drop", e=>{
      e.preventDefault();
      const de = arrastando, para = +el.dataset.i;
      arrastando = null;
      if(de === null || de === para) return;
      const [item] = DETS.splice(de, 1);
      DETS.splice(para, 0, item);
      pintaDets();          // renumera os indices
      scheduleSave();       // a ordem faz parte do estado salvo
    });
  });
}

/* ================= cor (grading) =================
   Mesmos parametros grading_* da UI simplificada e da aba dev. O worker aplica no
   FIM do pipeline (aplica_grading), depois do ultra — a regra do usuario: gerar →
   ultra → cor, nunca grading no meio. Defaults = o preset que ele usa. */
const COLS = [["grading_brightness","Brilho",-1,1,-0.05], ["grading_contrast","Contraste",-1,1,0.15],
  ["grading_saturation","Saturação",-1,1,0.10], ["grading_shadows","Sombras",-1,1,0.10],
  ["grading_highlights","Highlights",-1,1,0.05], ["grading_sharpness","Nitidez",0,1,0.15]];
let GRADING = {};
function montaCor(){
  $("colorBox").innerHTML = COLS.map(([k,lb,mn,mx,dv])=>`
    <div class="field"><div class="lbl">${lb} <span class="val" id="${k}_v">${dv.toFixed(2)}</span></div>
      <div class="slider"><input type="range" id="${k}" min="${mn}" max="${mx}" step="0.02" value="${dv}"></div>
    </div>`).join("");
  COLS.forEach(([k,,,,dv])=>{
    GRADING[k] = dv;
    const el = $(k);
    el.addEventListener("input", ()=>{
      GRADING[k] = +el.value; $(k+"_v").textContent = (+el.value).toFixed(2);
    });
  });
  $("colorReset").addEventListener("click", ()=> COLS.forEach(([k])=>{
    $(k).value = 0; $(k).dispatchEvent(new Event("input", {bubbles:true}));
  }));
}
function gradingAtivo(){
  // so manda o que esta REALMENTE mexido: o worker pula a etapa se tudo for ~0
  return Object.fromEntries(Object.entries(GRADING).filter(([,v])=> Math.abs(v) > 1e-3));
}

/* ================= centros manuais =================
   Caixas vindas do disco: escolher tag por tag em vez de digitar
   'person:girl,woman' — que era pedir p/ decorar a sintaxe e o lexico. */
function pintaCentros(){
  const nomes = Object.keys(CENTROS).sort();
  if(!nomes.length){ $("centrosBox").textContent = "nenhum especialista treinado"; return; }
  $("centrosBox").innerHTML = nomes.map(n=>{
    const d = CENTROS[n];
    const tags = (d.tags||[]).map(t=>
      `<label class="tag"><input type="checkbox" class="tg" data-c="${esc(n)}" value="${esc(t)}"> ${esc(t)}</label>`).join("");
    return `<div class="centro">
      <label class="chk"><input type="checkbox" class="ctrOn" value="${esc(n)}" data-global="${d.global?1:0}">
        <b>${esc(n)}</b>
        <span class="badge">${d.global ? "global" : "regional"}</span>
        <span class="badge dim">${esc(d.cluster||"")}</span></label>
      ${tags ? `<details class="tagwrap"><summary>tags — âncoras da máscara (opcional)</summary>
        <div class="tagbox">${tags}</div></details>` : ""}
    </div>`;
  }).join("");
}
function montaCentros(){
  const reg = [], glob = [];
  document.querySelectorAll(".ctrOn:checked").forEach(b=>{
    const n = b.value;
    if(b.dataset.global === "1"){ glob.push(n); return; }
    const tags = Array.from(document.querySelectorAll(`.tg[data-c="${n}"]:checked`)).map(t=>t.value);
    reg.push(n + ":" + tags.join(","));
  });
  return {centros: reg.join(";"), globais: glob.join(",")};
}
function marcaCentros(centrosStr, globaisStr){
  document.querySelectorAll(".ctrOn").forEach(x=> x.checked = false);
  document.querySelectorAll(".tg").forEach(x=> x.checked = false);
  (centrosStr||"").split(";").filter(Boolean).forEach(par=>{
    const [nome, palavras] = par.split(":");
    const box = document.querySelector(`.ctrOn[value="${nome}"]`);
    if(box){ box.checked = true; box.closest(".centro").querySelectorAll(".tagwrap").forEach(t=>t.open=true); }
    (palavras||"").split(",").filter(Boolean).forEach(t=>{
      const tg = document.querySelector(`.tg[data-c="${nome}"][value="${t.trim()}"]`);
      if(tg) tg.checked = true;
    });
  });
  (globaisStr||"").split(",").filter(Boolean).forEach(n=>{
    const box = document.querySelector(`.ctrOn[value="${n.trim()}"]`);
    if(box) box.checked = true;
  });
}

/* ================= custo previsto =================
   O que o disco e as medições dizem, não o que soa intuitivo:
   - VRAM residente NÃO cresce com o nº de centros (4,78 GB medido com 3, 5, 6 e 7
     nichos): eles ficam PAGINADOS na RAM e sobem à placa um por vez.
   - quem manda na VRAM é a RESOLUÇÃO e o comprimento do contexto (blocos do prompt).
   - o que cresce por centro é RAM (89 MB a 709 MB, varia 8x) e TEMPO (um forward
     por centro, por passo).
   Previsão de pico vem de MEDIÇÃO nesta máquina (tabela aprendida), não de fórmula. */
let VRAM = {tabela:{}, base_gb:4.78, total_gb:8.0};

function centrosAtivos(){
  // manual: o que está marcado. auto: o que o roteador escolheu por último.
  if(!$("auto").checked){
    return Array.from(document.querySelectorAll(".ctrOn:checked")).map(b=>b.value);
  }
  return (window.__ultimaRota || []);
}
function pxMax(){
  let px = baseW();
  CAMADAS.filter(c=>c.on && $("hires").checked).forEach(c=> px = Math.max(px, pxDaCamada(c)));
  if($("ultra").checked) px = Math.round(px * (+$("ultraScale").value || 1.5));
  return px;
}
function blocosPrompt(){
  // ~1,3 token por palavra no CLIP; 75 úteis por bloco. Estimativa grosseira até o
  // worker informar o número real (evento 'ctx'), que é o que a tabela indexa.
  const pal = ($("pos").value.trim().match(/\S+/g) || []).length;
  return Math.max(1, Math.ceil(pal * 1.3 / 75));
}
function atualizaCusto(){
  const nomes = centrosAtivos();
  const mb = nomes.reduce((s,n)=> s + ((CENTROS[n] && CENTROS[n].mb) || 0), 0);
  const px = pxMax(), bl = blocosPrompt();
  const medido = VRAM.tabela[`${px}|${bl}b`]
              || VRAM.tabela[`${px}|${Math.max(1,bl-1)}b`]
              || VRAM.tabela[`${px}|${bl+1}b`];
  let vram;
  if(medido){
    const folga = VRAM.total_gb - medido.gb;
    const cor = folga < 0.3 ? "err" : (folga < 0.9 ? "warn" : "ok");
    vram = `<b class="${cor}">pico medido ${medido.gb.toFixed(2)} GB</b> de ${VRAM.total_gb}`
         + ` (folga ${folga.toFixed(2)} GB)` + (medido.onde ? ` · maior em <i>${esc(medido.onde)}</i>` : "");
  } else {
    vram = `base ${VRAM.base_gb} GB + pico <b>não medido</b> nesta combinação`
         + ` (${px}px, ${bl} bloco${bl>1?"s":""}) — gere uma vez e eu passo a prever`;
  }
  const nCent = nomes.length;
  // no automatico, antes de rotear, nao HA centros conhecidos — dizer "0 centros"
  // pareceria zero custo em vez de "ainda nao sei"
  const ram = nCent
    ? `${(mb/1024).toFixed(2)} GB em ${nCent} centro${nCent===1?"":"s"}`
      + ` (${nomes.map(n=>`${esc(n)} ${(CENTROS[n]||{}).mb||"?"}MB`).join(", ")})`
    : ($("auto").checked ? `<i>desconhecida — clique em "👁 Ver o que ele escolhe"`
                         + ` para saber quais centros o prompt ativa</i>`
                         : `<i>nenhum centro marcado</i>`);
  // Quantos centros cabem RESIDENTES antes de empurrar o conjunto p/ o spill.
  // Medido: 4 centros residentes em 8GB = 7,6GB -> spill -> 53,9s/passo, contra
  // 15,0s/passo com eles paginados. O teto de centros nao e' um numero fixo; e'
  // esta conta. Acima dela ainda GERA, mas cada centro extra paga PCIe por passo.
  const folga = VRAM.total_gb - VRAM.base_gb - 1.6;      // 1,6 = piso do desktop
  const medioGb = nCent ? (mb / 1024) / nCent : 0.69;
  const cabem = Math.max(0, Math.floor(folga / Math.max(0.05, medioGb)));
  let alerta = "";
  if(nCent > cabem){
    alerta = `<div class="custo-l"><b class="warn">⚠️ ${nCent} centros × ~${medioGb.toFixed(2)} GB`
           + ` não cabem residentes (folga ~${folga.toFixed(1)} GB → ~${cabem}).</b>`
           + ` Os ${nCent - cabem} excedentes serão paginados: gera igual, mas cada um`
           + ` custa ~1 forward extra por passo.</div>`;
  }
  // ---------------------------------------------------------------- CAMADAS
  //
  // A conta que explica por que o GSMDE demora mais que um modelo comum, e que
  // ate' aqui o usuario nao tinha como fazer.
  //
  // Cada centro NAO e' uma LoRA empilhada no mesmo forward. Ele assume o UNet
  // inteiro por vez e produz a propria predicao sobre todo o latente; so' depois
  // as predicoes se combinam por mascara. Sao N passadas completas por passo, mais
  // a sonda que extrai as mascaras e a passada do backbone.
  //
  // Medido em 13/08: paginacao 0 ms, set_adapters 39 ms, cessao 40 ms — as passadas
  // respondem por ~98% do tempo. Entao o numero de camadas E' o custo.
  const nPas = nCent ? nCent + 2 : 1;           // centros + sonda de mascara + base
  const passos = +$("steps").value || 26;
  const camadas = nCent
    ? `<b>${nCent} centro${nCent===1?"":"s"} = ${nCent} camada${nCent===1?"":"s"} por passo</b>`
      + ` <span class="dim">(+2: sonda de máscara e base)</span>`
      + `<div class="custo-l dim">${passos} passos × ${nPas} = <b>${passos*nPas} passadas`
      + ` completas do modelo</b> — um modelo comum faz ${passos*2} com CFG em lote.`
      + ` É por isto que demora: <b>~${(nPas/2).toFixed(1)}× o trabalho</b>, não porque`
      + ` esteja mal otimizado.</div>`
    : `<i>depende dos centros que o prompt ativar</i>`;

  $("custo").innerHTML =
    `<div class="custo-l">CAMADAS &nbsp;${camadas}</div>`
  + `<div class="custo-l">VRAM &nbsp;${vram}</div>`
  + `<div class="custo-l">RAM &nbsp;&nbsp;${ram}</div>`
  + alerta
  + `<div class="custo-l dim">os centros não ocupam VRAM residente — são paginados da RAM;`
  + ` quem move o pico é resolução (${px}px) e contexto (${bl} bloco${bl>1?"s":""})</div>`;
}

/* ---------- visibilidade condicional ---------- */
function syncAuto(){
  const auto = $("auto").checked;
  $("gCentros").style.display = auto ? "none" : "";   // some no automatico
  if(!auto) $("gCentros").open = true;
}
function syncHires(){ $("hiresRows").style.display = $("hires").checked ? "" : "none"; }
function syncUltra(){ $("ultraRows").style.display = $("ultra").checked ? "" : "none"; }

/* ================= GALERIA DE LoRAs DA BIBLIOTECA ==========================

   O PAPEL DE CADA UMA — carona ou centro — vem MEDIDO, nao suposto:

     89 MB  como centro +7,16 s/it | como carona +3,62 s/it  (-49%)
     709 MB como centro  ~7  s/it  | como carona  ~7   s/it  ( -0%)

   Uma carona viaja DENTRO da passada de outro adaptador; um centro tem passada
   propria do UNet. A carona nao e' gratuita porque roda em TODAS as passadas —
   custa N x sobrecarga contra 1 passada inteira. Compensa enquanto for pequena.

   Por isso a miniatura mostra o papel: e' a informacao que muda o tempo de
   geracao, e o usuario decide vendo o custo, nao depois de esperar por ele.     */
let CATALOGO = null;
const LORAS_ESCOLHIDAS = new Map();   // id -> item

async function carregaCatalogoLoras(){
  if(CATALOGO) return CATALOGO;
  // ESPERA A PONTE. Abrir o grupo antes de o pywebview inicializar caia num
  // `return` silencioso, e como nao ha novo evento de toggle o painel ficava em
  // "carregando..." para sempre. Agora espera, e se desistir DIZ que desistiu:
  // um estado de carregamento eterno e' pior que um erro visivel.
  for(let i = 0; i < 40 && !api(); i++) await new Promise(r=>setTimeout(r, 150));
  if(!api()){
    $("loraResumo").innerHTML = `<b class="warn">ponte com o Python indisponível</b>`;
    return null;
  }
  try{
    const r = await api().gsmde_catalogo_loras();
    if(!r || r.__error__){
      $("loraResumo").innerHTML = `<b class="warn">${esc((r&&r.__error__)||"catálogo vazio")}</b>`
        + `<div class="dim">gere com: python D:/GSMDE/auto/gera_catalogo_loras.py</div>`;
      return null;
    }
    if(!r.loras || !r.loras.length){
      $("loraResumo").innerHTML = `<b class="warn">catálogo sem itens</b>`;
      return null;
    }
    CATALOGO = r;
    pintaGaleria();
    return r;
  }catch(e){
    $("loraResumo").innerHTML = `<b class="warn">falhou: ${esc(e && e.message || e)}</b>`;
    return null;
  }
}

function pintaGaleria(){
  if(!CATALOGO) return;
  const q = ($("loraBusca").value||"").toLowerCase().trim();
  const papel = $("loraPapel").value;
  const compat = ($("loraCompat")||{value:"todos"}).value;
  const todas = CATALOGO.loras || [];
  const vis = todas.filter(l=>{
    // 'ambos' aparece nos dois filtros: a faixa nao medida serve para os dois papeis
    // "ambos" casa com qualquer papel pedido: a faixa nao medida serve aos dois
    if(papel !== "todos" && l.papel !== papel && l.papel !== "ambos") return false;
    if(compat !== "todos" && l.compat !== compat) return false;
    if(!q) return true;
    return (l.id+" "+(l.titulo||"")+" "+(l.dominio||[]).join(" ")).toLowerCase().includes(q);
  });
  $("loraResumo").innerHTML =
      `<b>${vis.length}</b> de ${todas.length} mostradas`
    + ` <span class="dim">— carona custa ~metade de um centro; a faixa`
    + ` ${CATALOGO.teto_carona_mb}–${CATALOGO.piso_centro_mb||500} MB não foi medida</span>`;

  // TODAS sao desenhadas. O que era caro nao era o numero de linhas e sim buscar
  // 389 miniaturas pela ponte; agora elas carregam so' quando entram na tela.
  $("loraGrade").innerHTML = vis.map(l=>{
    const sel = LORAS_ESCOLHIDAS.has(l.id) ? " sel" : "";
    const pc = {carona:"car", ambos:"amb", centro:"cen"}[l.papel] || "cen";
    const cc = {sim:"ok", nao:"no", desconhecida:"dq"}[l.compat] || "dq";
    const ct = {sim:"compatível", nao:"incompatível", desconhecida:"família ?"}[l.compat];
    const semTag = (l.dominio||[]).length ? "" : `<i class="lr-warn" title="sem etiquetas">sem tags</i>`;
    return `<div class="lr${sel}" data-id="${esc(l.id)}">`
         + `<div class="lr-img" data-thumb="${esc(l.thumb||"")}"></div>`
         + `<div class="lr-txt">`
         +   `<div class="lr-nome">${esc(l.titulo||l.id)}</div>`
         +   `<div class="lr-tags">`
         +     `<span class="lr-tag ${pc}">${l.papel}</span>`
         +     `<span class="lr-tag ${cc}">${ct}</span>`
         +     semTag
         +   `</div>`
         +   `<div class="lr-mb">${l.mb} MB${l.posto?" · posto "+l.posto:""}`
         +     `${(l.dominio||[]).length?" · "+esc(l.dominio.slice(0,4).join(", ")):""}</div>`
         + `</div></div>`;
  }).join("") || `<div class="mini dim">nada com esse filtro</div>`;
  observaThumbs();
  pintaEscolhidas();
}

/* Miniatura sob demanda: so' quando a linha entra na tela. Sem isto, mostrar as
   389 dispararia 389 chamadas de ponte de uma vez e travaria a rolagem. */
let OBS_THUMB = null;
function observaThumbs(){
  if(!OBS_THUMB){
    OBS_THUMB = new IntersectionObserver(async (ents)=>{
      for(const e of ents){
        if(!e.isIntersecting) continue;
        const el = e.target;
        OBS_THUMB.unobserve(el);
        if(el.dataset.pronto || !el.dataset.thumb){ el.classList.add("vazia"); continue; }
        el.dataset.pronto = "1";
        try{
          const d = await api().gsmde_thumb(el.dataset.thumb);
          if(d) el.style.backgroundImage = `url('${d}')`; else el.classList.add("vazia");
        }catch(err){ el.classList.add("vazia"); }
      }
    }, {root: $("loraGrade"), rootMargin: "200px"});
  }
  document.querySelectorAll(".lr-img:not([data-pronto])").forEach(el=>OBS_THUMB.observe(el));
}

function pintaEscolhidas(){
  const n = LORAS_ESCOLHIDAS.size;
  if(!n){ $("loraEscolhidas").innerHTML = ""; return; }
  const itens = Array.from(LORAS_ESCOLHIDAS.values());
  const car = itens.filter(l=>l.papel==="carona").length;
  $("loraEscolhidas").innerHTML =
      `<b>${n} escolhida${n>1?"s":""}</b>: ${itens.map(l=>esc(l.titulo||l.id)).join(", ")}`
    + `<div class="dim">${car} como carona (sem passada extra) · ${n-car} como centro`
    + ` (+1 passada cada)</div>`;
}

/* Clique: insere as tags de dominio no prompt e marca a LoRA.
   O roteador le o prompt DEPOIS, entao inserir a tag e' o que faz o centro
   correspondente ser convocado — e a LoRA escolhida entra como carona por cima,
   em vez de virar mais um centro. */
function alternaLora(id){
  if(!CATALOGO) return;
  const l = (CATALOGO.loras||[]).find(x=>x.id===id);
  if(!l) return;
  const ta = $("pos");
  const marca = (l.dominio||[]).slice(0,3).join(", ");
  if(LORAS_ESCOLHIDAS.has(id)){
    LORAS_ESCOLHIDAS.delete(id);
    if(marca) ta.value = ta.value.replace(marca, "").replace(/,\s*,/g, ",")
                                 .replace(/^\s*,\s*/, "").trim();
  }else{
    LORAS_ESCOLHIDAS.set(id, l);
    if(marca && !ta.value.includes(marca))
      ta.value = (ta.value.trim() ? ta.value.trim().replace(/,\s*$/,"") + ", " : "") + marca;
  }
  ta.dispatchEvent(new Event("input", {bubbles:true}));
  pintaGaleria();
  atualizaCusto();
}

/* Popup de dominio no hover: mostra as etiquetas REAIS do treino da LoRA,
   lidas de ss_tag_frequency. Nao e' descricao editorial — e' o que o autor
   mostrou ao modelo. */
function montaGaleriaLoras(){
  const grade = $("loraGrade");
  if(!grade) return;
  let pop = null;
  grade.addEventListener("mouseover", (e)=>{
    const c = e.target.closest(".lr"); if(!c) return;
    const l = (CATALOGO?.loras||[]).find(x=>x.id===c.dataset.id); if(!l) return;
    if(!pop){ pop = document.createElement("div"); pop.className = "lr-pop";
              document.body.appendChild(pop); }
    const dom = (l.dominio||[]);
    const cc = {sim:"ok", nao:"no", desconhecida:"dq"}[l.compat] || "dq";
    pop.innerHTML = `<b>${esc(l.titulo||l.id)}</b>`
      + `<div class="dim">${l.mb} MB${l.posto?" · posto "+l.posto:""}`
      + `${l.imagens?" · "+l.imagens+" imagens de treino":""}</div>`
      + (dom.length ? `<div class="lr-dom">${dom.map(t=>`<i>${esc(t)}</i>`).join("")}</div>`
                    : "")
      + `<div class="lr-por"><b>papel:</b> ${esc(l.motivo_papel||l.papel)}</div>`
      + `<div class="lr-por ${cc}"><b>compatibilidade:</b> ${esc(l.motivo_compat||"—")}</div>`
      + (dom.length ? "" : `<div class="lr-por no"><b>roteamento:</b> ${esc(l.motivo_dominio||"")}</div>`);
    const r = c.getBoundingClientRect();
    pop.style.display = "block";
    pop.style.left = Math.min(window.innerWidth-320, r.right+8) + "px";
    pop.style.top  = Math.min(window.innerHeight-190, r.top) + "px";
  });
  grade.addEventListener("mouseout", (e)=>{
    if(pop && !e.relatedTarget?.closest?.(".lr")) pop.style.display = "none";
  });
  grade.addEventListener("click", (e)=>{
    const c = e.target.closest(".lr"); if(c) alternaLora(c.dataset.id);
  });
  $("loraBusca").addEventListener("input", pintaGaleria);
  $("loraPapel").addEventListener("change", pintaGaleria);
  $("loraCompat").addEventListener("change", pintaGaleria);
  $("loraRecarrega").addEventListener("click", async ()=>{
    const b = $("loraRecarrega");
    b.disabled = true; b.textContent = "…";
    $("loraResumo").textContent = "relendo a biblioteca do disco…";
    try{
      // regera o catalogo no Python: le os .safetensors de novo, entao pega
      // LoRAs adicionadas depois que a UI abriu
      const r = await api().gsmde_recatalogar();
      if(r && r.__error__) throw new Error(r.__error__);
      CATALOGO = null;
      document.querySelectorAll(".lr-img").forEach(e=>delete e.dataset.pronto);
      await carregaCatalogoLoras();
    }catch(e){
      $("loraResumo").innerHTML = `<b class="warn">${esc(e && e.message || e)}</b>`;
    }finally{ b.disabled = false; b.textContent = "↻"; }
  });
  $("gLoras").addEventListener("toggle", ()=>{ if($("gLoras").open) carregaCatalogoLoras(); });
  // o grupo pode ja' estar aberto (estado salvo da sessao anterior): sem isto o
  // toggle nunca dispara e o catalogo nunca carrega
  if($("gLoras").open) carregaCatalogoLoras();
}

/* ---------- backbone: qual UNet vai para a placa ----------------------------

   Medido em 12/08 numa bancada de 54 geracoes a 1024 (docs/relatorio.md): a UNet
   ORIGINAL e' a unica que transborda para a RAM. Reduzir nao e' so' economia — e' o
   que tira a geracao da zona de despejo nesta placa de 8 GB.

   O custo e' que nem todo modulo de LoRA encontra alvo; os que somem estao no nivel
   profundo, e o efeito aparece em detalhe de material e objeto pequeno, nao em
   pessoa. Por isso o numero fica escrito na propria opcao: a escolha e' do usuario e
   ela tem um preco que ele precisa ver antes de escolher.                        */
const FILA_TODAS = [];      // quando "Todas" esta escolhido, as variantes pendentes
let varianteEmCurso = null; // a que esta rodando agora (rotula a barra de estado)

function varianteAtual(){
  const v = $("backboneVar") ? $("backboneVar").value : "original";
  if(v !== "__todas__") return v;
  // no modo "Todas" quem manda e' a fila; se ela secou, a rodada e' da primeira
  return varianteEmCurso || todasAsVariantes()[0];
}

function todasAsVariantes(){
  return Array.from($("backboneVar").options)
              .map(o=>o.value).filter(v=>v !== "__todas__");
}

function syncBackboneVar(){
  const s = $("backboneVar"), info = $("backboneVarInfo");
  if(!s || !info) return;
  if(s.value === "__todas__"){
    const n = todasAsVariantes().length;
    info.textContent = `${n} gerações, mesma seed e mesmo prompt em cada modelo — ` +
                       `serve para comparar, não para produzir.`;
  } else if(s.value === "original"){
    info.textContent = "a única que transborda para a RAM a 1024 (0,30 GB medidos); " +
                       "em compensação, todas as LoRAs encaixam.";
  } else {
    const t = s.options[s.selectedIndex].textContent;
    const pct = (t.match(/(\d+)%/) || [])[1];
    info.textContent = `sem transbordo. ${pct}% dos módulos de LoRA encaixam — ` +
                       `o que falta pesa em material e objeto pequeno, não em pessoa.`;
  }
}

/* ---------- monta a config e dispara ---------- */
function buildCfg(){
  if($("seedRand").checked) $("seed").value = Math.floor(Math.random()*2147483647);
  const man = montaCentros();
  const infinita = $("infinita").checked;
  return {
    auto: $("auto").checked, max_centros: +$("maxCentros").value || 6,
    prompt: $("pos").value, negative: $("neg").value,
    centros: man.centros, globais: man.globais,
    escala:+$("escala").value, gw:+$("gw").value, cfg:+$("cfg").value,
    steps:+$("steps").value||26, seed:+$("seed").value||1234,
    width: baseW(), height: baseH(), size: baseW(),   // 'size' = fallback/metadados
    backbone_assert:+$("backbone").value,
    variante: varianteAtual(),
    weighted:$("weighted").checked, center_focus:$("centerFocus").checked,
    focus_w:+$("focusW").value||1.15, context_w:+$("contextW").value||0.9,
    mask_every:+$("maskEvery").value||4, yield_ms:+$("yieldMs").value||40,
    paginacao:$("paginacao").value, vram_reserva:+$("vramReserva").value||0,
    offload_base:$("offloadBase").value, offload_teto:+$("offloadTeto").value||0,
    refinar:$("hires").checked, camadas:strCamadas(), upscaler:$("upscaler").value,
    detailers: DETS.map(d=>({model:d.model, prompt:d.prompt||"",
                             denoise:d.dn, steps:d.st})),
    // Medido em 17/08: sem máscara com buracos o detailer redesenha o que não é
    // alvo (o nariz entre os olhos); sem a regra do par, repintar um olho só
    // deixa as duas íris de cores diferentes. Padrão ligado nos dois.
    det_mascarado:$("detMascarado").checked, det_par:$("detPar").checked,
    det_lado_min:+$("detLadoMin").value||256, det_crop:+$("detCrop").value||1024,
    ultra:$("ultra").checked, ultra_halo:$("ultra").checked,
    ultra_scale:+$("ultraScale").value, ultra_core:+$("ultraTile").value,
    ultra_pad:+$("ultraPad").value, ultra_overlap:+$("ultraOv").value,
    ultra_passes:+$("ultraPasses").value, ultra_final:+$("ultraFinal").value,
    ultra_denoise:+$("ultraDenoise").value, ultra_steps:+$("ultraSteps").value,
    grading: gradingAtivo(),
    preview_every: $("preview").checked ? (+$("previewEvery").value||3) : 0,
    // 'manter' protege as imagens no loop infinito; 'limpar_intermed' e' faxina
    // EXPLICITA — a pasta final nunca e' tocada pelo launcher.
    manter: infinita,
    limpar_intermed: $("limpar").checked && !infinita,
  };
}

async function gerar(){
  if(generating || !api()) return;
  // O UNICO bloqueio e' o teto do sistema. Overload de buffer (niveis 1 e 2)
  // incomoda mas deixa gerar — a escolha do numero de centros e' do usuario.
  const r = await ramMede();
  if(r && r.critico){
    avisa(`⛔ RAM do sistema em ${Math.round(r.uso*100)}% (bloqueio em ` +
          `${Math.round(r.teto_critico*100)}%). Feche algum programa — gerar agora ` +
          `jogaria a máquina em swap e cada passo levaria minutos.`);
    return;
  }
  pararLoop = false; rodada = 0;

  // MODO "TODAS": mesma seed, mesmo prompt, um modelo por vez.
  //
  // Sequencial e nao paralelo por decisao, nao por preguica: duas cargas na mesma
  // placa ja derrubaram o driver para o generico da Microsoft duas vezes neste
  // projeto. E a seed e' fixada AQUI, antes da fila, senao "seed aleatoria" sortearia
  // uma por rodada e o comparativo compararia seeds em vez de modelos.
  FILA_TODAS.length = 0;
  varianteEmCurso = null;
  if($("backboneVar") && $("backboneVar").value === "__todas__"){
    if($("seedRand").checked){
      $("seed").value = Math.floor(Math.random()*2147483647);
      $("seedRand").checked = false;   // visivel: o usuario ve por que parou de sortear
      avisa("Modo Todas: seed fixada em " + $("seed").value +
            " para que o comparativo compare modelos, não seeds.");
    }
    FILA_TODAS.push(...todasAsVariantes());
    varianteEmCurso = FILA_TODAS.shift();
  }
  await umaRodada();
}

async function umaRodada(){
  const cfg = buildCfg();
  const infinita = $("infinita").checked;
  setBusy(true, "enviando job…"); $("pfill").style.width="0%"; pvSeen=-1; etapasSeen=0;
  const r = await api().gsmde_start(cfg);
  if(r && r.__error__){ pararLoop = true; setBusy(false, "erro"); showLog(r.__error__); return; }
  if(r && r.aviso) avisa(r.aviso);      // gera assim mesmo, mas o usuario fica sabendo
  rodada++;
  setBusy(true, infinita ? `gerando #${rodada} (seed ${cfg.seed})` : "gerando");
  clearInterval(pollTimer);
  pollTimer = setInterval(()=> poll(infinita, cfg.seed), 1200);
}

async function poll(infinita, seed){
  const st = await api().gsmde_status(); if(!st) return;
  const pre = infinita ? `#${rodada} · ` : "";
  if(st.passo) $("stage").textContent = pre + st.passo;
  monitorVivo(st);      // RAM/VRAM reais ao lado do estagio
  // preview ao vivo enquanto nenhuma etapa final chegou
  if(st.preview && st.preview_i !== pvSeen && (!st.etapas || !st.etapas.length)){
    pvSeen = st.preview_i; showImg(st.preview);
    $("stage").textContent = pre + "preview " + st.preview_i;
  }
  // so redesenha quando uma etapa NOVA chega: repintar o mesmo dataurl a cada
  // 1,2s custa decode a toa e derrubaria o log aberto por cima do visor.
  if(st.etapas && st.etapas.length && st.etapas.length !== etapasSeen){
    etapasSeen = st.etapas.length;
    showImg(st.etapas[st.etapas.length-1].dataurl);
    $("pfill").style.width = "100%";
    $("etapas").textContent = (infinita ? `rodada ${rodada} (seed ${seed}) · ` : "")
      + "etapas: " + st.etapas.map(e=>e.nome).join(" → ");
  }
  if(st.final && !$("etapas").textContent.includes("final salva"))
    $("etapas").textContent += "  ·  final salva";
  if(!st.running){
    clearInterval(pollTimer); pollTimer=null;
    if(st.rc!==0){
      pararLoop = true; setBusy(false, "falhou (rc="+st.rc+")");
      // as etapas que DERAM certo continuam no disco: nada e' apagado no erro.
      showLog("rc=" + st.rc + "\n\n" + (st.log || "(sem log)"));
      return;
    }
    $("pfill").style.width="100%";
    // a rodada acabou de gravar o pico medido: recarrega a tabela p/ o painel de
    // custo passar a prever esta combinacao sem precisar reabrir a UI
    api().gsmde_vram_tabela().then(v=>{
      if(v && !v.__error__){ VRAM = v; atualizaCusto(); }
    }).catch(()=>{});
    // le a caixa AGORA, nao no inicio da rodada: desmarcar "infinita" no meio de
    // uma geracao tem que encerrar o loop quando ela terminar.
    // a fila do modo "Todas" vem ANTES da infinita: sao coisas diferentes, e quem
    // pediu um comparativo quer o comparativo inteiro, nao um loop na primeira
    if(FILA_TODAS.length && !pararLoop){
      varianteEmCurso = FILA_TODAS.shift();
      setBusy(true, `comparativo: ${varianteEmCurso} ` +
                    `(faltam ${FILA_TODAS.length})`);
      setTimeout(umaRodada, 400);
    } else if(varianteEmCurso && !FILA_TODAS.length &&
              $("backboneVar").value === "__todas__"){
      varianteEmCurso = null;
      setBusy(false, "comparativo completo");
    } else if($("infinita").checked && !pararLoop){
      setBusy(true, `rodada ${rodada} pronta — indo p/ a próxima`);
      setTimeout(umaRodada, 400);
    } else {
      setBusy(false, rodada > 1 ? `concluído — ${rodada} rodadas` : "concluído");
    }
  }
}

async function parar(){
  pararLoop = true;
  if(pollTimer){ clearInterval(pollTimer); pollTimer=null; }
  if(api()) await api().gsmde_stop();
  setBusy(false, "parado");
}

async function verCentros(){
  if(!api()) return;
  const r = await api().gsmde_route($("pos").value, +$("maxCentros").value||6);
  if(!r || r.__error__){ $("routeOut").textContent = (r&&r.__error__) || "—"; return; }
  const lista = (r.centros||"").split(";").filter(Boolean).map(x=>x.split(":")[0]);
  window.__ultimaRota = lista.concat((r.globais||"").split(",").filter(Boolean));
  // o numero de camadas vem JUNTO da lista de centros: e' o momento em que o
  // usuario descobre quantos especialistas o prompt convocou, e portanto o momento
  // certo para dizer quanto trabalho isso significa
  const nc = window.__ultimaRota.length;
  const ps = +$("steps").value || 26;
  $("routeOut").innerHTML = "<b>centros:</b> " + (lista.join(", ")||"(base pura)")
    + (r.globais ? " · <b>globais:</b> "+esc(r.globais) : "")
    + (nc ? `<div class="mini" style="margin-top:4px">≡ <b>${nc} camadas</b> por passo`
            + ` · ${ps} passos × ${nc+2} = <b>${ps*(nc+2)} passadas</b> do modelo`
            + ` <span class="dim">(um modelo comum faria ${ps*2})</span></div>` : "");
  atualizaCusto();
  return r;
}

/* Imagens geradas antes desta versao guardaram o ROTULO ("RealESRGAN anime 6B");
   o motor so entende a chave curta. Sem isto, reciclar uma imagem antiga deixava
   o select intocado e em silencio. Espelha chave_upscaler() do worker. */
function chaveUpscaler(v){
  const s = String(v||"").trim(), b = s.toLowerCase();
  if(["anime","real","sharp"].includes(s)) return s;
  if(b.includes("anime")) return "anime";
  if(b.includes("sharp")) return "sharp";
  if(b.includes("esrgan") || b.includes("real")) return "real";
  return "anime";
}

/* ---------- reciclar: le os parametros gravados no PNG ---------- */
async function reciclar(){
  if(!api()) return;
  const d = await api().gsmde_pick_meta();
  if(!d) return;
  if(d.__error__){ toast(d.__error__); showLog(d.__error__); return; }
  const set = (id, v)=>{ if(v !== undefined && v !== null && v !== ""){
    $(id).value = v; $(id).dispatchEvent(new Event("input")); } };
  set("pos", d.prompt); set("neg", d.negative);
  set("steps", d.steps); set("cfg", d.cfg); set("seed", d.seed);
  set("largura", d.width || d.size); set("altura", d.height || d.size);
  atualizaProp();
  $("seedRand").checked = false;                 // reciclar e' p/ repetir a imagem
  if(d._fonte !== "gsmde"){
    toast(`${d._arquivo}: metadados A1111 — prompt/seed/steps recuperados`);
    return;
  }
  set("escala", d.escala); set("gw", d.gw);
  if(d.backbone_assert !== undefined && d.backbone_assert !== null) set("backbone", d.backbone_assert);
  if(d.max_centros) set("maxCentros", d.max_centros);
  if(d.focus_w) set("focusW", d.focus_w);
  if(d.context_w) set("contextW", d.context_w);
  if(d.upscaler) $("upscaler").value = chaveUpscaler(d.upscaler);
  if(d.weighted !== undefined) $("weighted").checked = !!d.weighted;
  if(d.center_focus !== undefined) $("centerFocus").checked = !!d.center_focus;
  if(d.auto !== undefined){ $("auto").checked = !!d.auto; syncAuto(); }
  marcaCentros(d.centros, d.globais);
  if(d.camadas){
    // metadados guardam px absoluto; a UI trabalha em ganho sobre a largura base
    const lb = +(d.width || d.size) || baseW();
    CAMADAS = d.camadas.split(",").filter(Boolean).map(x=>{
      const [px,dn,st] = x.split(":");
      return {g:+(+px / lb).toFixed(2), dn:+dn, st:+st, on:true};
    });
    pintaCamadas();
  }
  $("hires").checked = !!d.camadas; syncHires();
  DETS = (d.detailers||[]).map(x=>({model:x.model, prompt:x.prompt||"",
                                    dn:+x.denoise||0.3, st:+x.steps||20}));
  pintaDets();
  // `!== undefined` e nao `||`: com `||`, desmarcar a caixa e salvar traria ela
  // de volta marcada na proxima sessao, porque false cairia no padrao.
  if(d.det_mascarado !== undefined) $("detMascarado").checked = !!d.det_mascarado;
  if(d.det_par !== undefined) $("detPar").checked = !!d.det_par;
  if(d.det_lado_min) set("detLadoMin", d.det_lado_min);
  if(d.det_crop) set("detCrop", d.det_crop);
  if(d.ultra !== undefined){ $("ultra").checked = !!d.ultra; syncUltra(); }
  if(d.ultra_scale) set("ultraScale", d.ultra_scale);
  if(d.ultra_core) set("ultraTile", d.ultra_core);
  if(d.ultra_pad) set("ultraPad", d.ultra_pad);
  if(d.ultra_overlap) set("ultraOv", d.ultra_overlap);
  if(d.ultra_passes) set("ultraPasses", d.ultra_passes);
  if(d.ultra_denoise) set("ultraDenoise", d.ultra_denoise);
  if(d.ultra_steps) set("ultraSteps", d.ultra_steps);
  if(d.ultra_final) set("ultraFinal", d.ultra_final);
  if(d.grading) COLS.forEach(([k])=>{           // cor tambem volta do PNG
    if(d.grading[k] !== undefined) set(k, d.grading[k]);
  });
  toast(`reciclado de ${d._arquivo} (${d._size||""})`);
}

/* ================= persistencia da sessao =================
   Fechou e abriu = continua onde parou. Guarda em prefs.json (tools['gsmde-studio']),
   a mesma gaveta que as ferramentas usam. Diferente da aba dev, aqui tambem entram
   camadas, alvos do detailer e os centros marcados — que sao DOM criado na hora e
   ficariam de fora de um coletor que so varre inputs. */
const CHAVE_ESTADO = "gsmde-studio";
let saveTimer = null, restaurando = false;

function collectState(){
  const inputs = {};
  document.querySelectorAll("#left input[id], #left select[id], #left textarea[id]")
    .forEach(el=>{ inputs[el.id] = (el.type === "checkbox") ? el.checked : el.value; });
  return {
    inputs, camadas: CAMADAS, dets: DETS, hint: hintOn,
    centros: montaCentros(),
    abertos: Array.from(document.querySelectorAll("#groups > details")).map(d=>d.open),
  };
}
function scheduleSave(){
  if(restaurando || !api()) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(()=>{
    try{ api().save_tool_state(CHAVE_ESTADO, collectState()); }catch(_){}
  }, 400);
}
function applyState(s){
  if(!s) return;
  restaurando = true;
  try{
    for(const [k,v] of Object.entries(s.inputs || {})){
      const el = $(k); if(!el) continue;
      if(el.type === "checkbox") el.checked = !!v; else el.value = v;
      el.dispatchEvent(new Event("input"));
    }
    if(Array.isArray(s.camadas) && s.camadas.length){
      // estado gravado antes do ganho guardava px absoluto: converte na largura atual
      CAMADAS = s.camadas.map(c=> c.g !== undefined ? c
        : {g:+(+c.px / baseW()).toFixed(2), dn:c.dn, st:c.st, on:c.on !== false});
    }
    if(Array.isArray(s.dets)){ DETS = s.dets; pintaDets(); }
    if(s.hint !== undefined) setHint(s.hint);
    atualizaProp();      // recalcula proporcao e repinta as camadas com a largura nova
    if(s.centros) marcaCentros(s.centros.centros, s.centros.globais);
    if(Array.isArray(s.abertos))
      document.querySelectorAll("#groups > details").forEach((d,i)=>{
        if(s.abertos[i] !== undefined) d.open = s.abertos[i]; });
    syncAuto(); syncHires(); syncUltra();
    $("seed").disabled = $("seedRand").checked;
  } finally { restaurando = false; }
}

/* ================= i2i: mascara manual =================
   Duas telas empilhadas: a imagem embaixo, a mascara por cima em vermelho. A
   mascara e' desenhada em RESOLUCAO NATIVA da imagem (o canvas so e' exibido
   reduzido por CSS) — reescalar a mascara depois introduziria erro de borda
   justamente onde a costura precisa ser invisivel. */
let I2I = {img:null, w:0, h:0, pintando:false, apagar:false};

function i2iCarrega(dataurl){
  const im = new Image();
  im.onload = ()=>{
    I2I.img = im; I2I.w = im.naturalWidth; I2I.h = im.naturalHeight;
    const base = $("i2iBase"), mask = $("i2iMask");
    base.width = mask.width = I2I.w;
    base.height = mask.height = I2I.h;
    base.getContext("2d").drawImage(im, 0, 0);
    mask.getContext("2d").clearRect(0, 0, I2I.w, I2I.h);
    $("i2iPalco").style.display = "";
    $("viewEmpty").style.display = "none";
    $("viewImg").style.display = "none";
    $("i2iInfo").textContent = `${I2I.w}×${I2I.h} px`;
    i2iCusto();
  };
  im.src = dataurl;
}

function i2iPonto(ev){
  const mask = $("i2iMask"), r = mask.getBoundingClientRect();
  // do espaco da TELA p/ o espaco da IMAGEM (o canvas aparece reduzido)
  const x = (ev.clientX - r.left) * (I2I.w / r.width);
  const y = (ev.clientY - r.top) * (I2I.h / r.height);
  const g = mask.getContext("2d");
  const raio = (+$("i2iBrush").value || 60) / 2 * (I2I.w / r.width);
  g.globalCompositeOperation = I2I.apagar ? "destination-out" : "source-over";
  g.fillStyle = "rgba(220,60,60,0.55)";
  g.beginPath(); g.arc(x, y, raio, 0, Math.PI * 2); g.fill();
}

function i2iMascaraPB(){
  // a mascara vai p/ o motor em preto e branco: branco = gerar do zero
  const src = $("i2iMask").getContext("2d").getImageData(0, 0, I2I.w, I2I.h);
  const out = document.createElement("canvas");
  out.width = I2I.w; out.height = I2I.h;
  const d = out.getContext("2d").createImageData(I2I.w, I2I.h);
  for(let i = 0; i < src.data.length; i += 4){
    const v = src.data[i+3] > 8 ? 255 : 0;      // alfa pintado -> branco
    d.data[i] = d.data[i+1] = d.data[i+2] = v; d.data[i+3] = 255;
  }
  out.getContext("2d").putImageData(d, 0, 0);
  return out.toDataURL("image/png");
}

function i2iArea(){
  if(!I2I.img) return null;
  const s = $("i2iMask").getContext("2d").getImageData(0, 0, I2I.w, I2I.h).data;
  let x0 = I2I.w, y0 = I2I.h, x1 = -1, y1 = -1, n = 0;
  for(let y = 0; y < I2I.h; y++) for(let x = 0; x < I2I.w; x++){
    if(s[(y*I2I.w + x)*4 + 3] > 8){
      n++; if(x < x0) x0 = x; if(x > x1) x1 = x; if(y < y0) y0 = y; if(y > y1) y1 = y;
    }
  }
  if(n === 0) return null;
  const folga = (+$("i2iHalo").value || 128) + (+$("i2iBlend").value || 48);
  const cw = Math.min(I2I.w, (x1 - x0 + 1) + folga*2);
  const ch = Math.min(I2I.h, (y1 - y0 + 1) + folga*2);
  return {cw, ch, mp: cw*ch/1e6, pintado: n};
}

function i2iCusto(){
  const a = i2iArea();
  if(!a){ $("i2iCusto").innerHTML = "<i>pinte a área que deve ser gerada</i>"; return; }
  const cabe = a.mp <= 1.5;
  $("i2iCusto").innerHTML =
    `região a processar: <b>${a.cw}×${a.ch}</b> (${a.mp.toFixed(2)} MP)`
  + ` — máscara + halo + blending`
  + (cabe ? "" : `<br><b class="err">acima do limite de 1,5 MP</b>: pinte menor ou reduza o halo`
             + ` (sem tiling e sem reescala não há como diluir sem criar costura)`);
}

async function i2iGerar(){
  if(!api() || !I2I.img) return;
  const a = i2iArea();
  if(!a){ toast("pinte a máscara primeiro"); return; }
  if($("i2iSeedRand").checked) $("i2iSeed").value = Math.floor(Math.random()*2147483647);
  const base = $("i2iBase").toDataURL("image/png");
  setBusy(true, "i2i: gerando na máscara…");
  etapasSeen = 0; pvSeen = -1;
  const r = await api().gsmde_start({
    modo: "inpaint",
    prompt: $("i2iPos").value, negative: $("neg").value,
    auto: true, max_centros: +$("maxCentros").value || 6,
    init_image: base, mask_image: i2iMascaraPB(),
    halo: +$("i2iHalo").value || 128, blend: +$("i2iBlend").value || 48,
    ctx_global: +$("i2iCtxGlobal").value || 0,
    steps: +$("i2iSteps").value || 30, seed: +$("i2iSeed").value || 7,
    cfg: +$("cfg").value, escala: +$("escala").value, gw: +$("gw").value,
    backbone_assert: +$("backbone").value,
    variante: varianteAtual(),
    paginacao: $("paginacao").value, vram_reserva: +$("vramReserva").value || 0,
    offload_base: $("offloadBase").value, offload_teto: +$("offloadTeto").value || 0,
    weighted: $("weighted").checked, manter: true,
  });
  if(r && r.__error__){ setBusy(false, "erro"); showLog(r.__error__); return; }
  clearInterval(pollTimer);
  pollTimer = setInterval(()=> poll(false, +$("i2iSeed").value || 0), 1200);
}

function montaI2i(){
  $("i2iFile").addEventListener("change", (e)=>{
    const f = e.target.files[0]; if(!f) return;
    const rd = new FileReader();
    rd.onload = ()=> i2iCarrega(rd.result);
    rd.readAsDataURL(f);
  });
  const mask = $("i2iMask");
  mask.addEventListener("mousedown", (e)=>{ I2I.pintando = true; i2iPonto(e); });
  mask.addEventListener("mousemove", (e)=>{ if(I2I.pintando) i2iPonto(e); });
  ["mouseup","mouseleave"].forEach(ev=> mask.addEventListener(ev, ()=>{
    if(I2I.pintando){ I2I.pintando = false; i2iCusto(); }
  }));
  $("i2iLimpar").addEventListener("click", ()=>{
    if(I2I.img) $("i2iMask").getContext("2d").clearRect(0, 0, I2I.w, I2I.h);
    i2iCusto();
  });
  const modo = (apagar)=>{
    I2I.apagar = apagar;
    $("i2iModoPintar").classList.toggle("on", !apagar);
    $("i2iModoApagar").classList.toggle("on", apagar);
  };
  $("i2iModoPintar").addEventListener("click", ()=> modo(false));
  $("i2iModoApagar").addEventListener("click", ()=> modo(true));
  ["i2iBrush","i2iHalo","i2iBlend","i2iCtxGlobal"].forEach((id)=>{
    const el = $(id), out = $(id+"_v");
    const upd = ()=>{
      if(out) out.textContent = el.value;
      if(id === "i2iHalo" || id === "i2iBlend") i2iCusto();  // mudam a area processada
    };
    el.addEventListener("input", upd); upd();
  });
  $("i2iGerar").addEventListener("click", i2iGerar);
}

/* ================= abas =================
   A aba de configuracao SUBSTITUI a coluna da esquerda (prompts + ajustes da
   imagem) pelos ajustes de BACKEND. Os controles nao sao duplicados: os mesmos
   elementos sao MOVIDOS p/ ca — assim buildCfg/collectState continuam achando os
   ids de sempre e nao existe estado em dois lugares. */
function montaAbas(){
  const mover = (id, destino) => {
    const el = $(id), alvo = $(destino);
    const campo = el && el.closest(".field");
    if(campo && alvo) alvo.appendChild(campo);
  };
  mover("paginacao", "slotPaginacao"); mover("vramReserva", "slotPaginacao");
  mover("offloadBase", "slotOffload"); mover("offloadTeto", "slotOffload");
  mover("maskEvery", "slotExec");      mover("yieldMs", "slotExec");

  const trocar = (aba)=>{
    document.querySelectorAll(".aba").forEach(b=>
      b.classList.toggle("ativa", b.dataset.aba === aba));
    const gerar = aba === "gerar";
    $("promptZone").style.display = gerar ? "" : "none";
    $("groups").style.display     = gerar ? "" : "none";
    document.querySelectorAll("#left > .actions").forEach(e=>
      e.style.display = gerar ? "" : "none");
    $("colConfig").style.display = aba === "config" ? "" : "none";
    $("colAgenda").style.display  = aba === "agenda"  ? "" : "none";
    $("colModelos").style.display = aba === "modelos" ? "" : "none";
    $("colI2i").style.display    = aba === "i2i"    ? "" : "none";
    // o palco da mascara so aparece no i2i; o visor normal volta nas outras abas
    $("i2iPalco").style.display  = (aba === "i2i" && I2I.img) ? "" : "none";
    // as contas (CivitAI e HF) vivem na Configuracao; a HF morava na aba
    // Modelos e ficou sem quem a atualizasse quando mudou de lugar.
    if(aba === "config"){ atualizaCfgVram(); mdStatus(); cvStatus(); ramMede(); }
    if(aba === "agenda") agCarrega();
    if(aba === "modelos") mdStatus();
    if(aba === "i2i") i2iCusto();
  };
  document.querySelectorAll(".aba").forEach(b=>
    b.addEventListener("click", ()=> trocar(b.dataset.aba)));
  trocar("gerar");
}

async function atualizaCfgVram(){
  if(!api()) return;
  const v = await api().gsmde_vram_tabela();
  if(v && !v.__error__) VRAM = v;
  const linhas = Object.entries(VRAM.tabela || {}).map(([k, d])=>{
    const [px, bl] = k.split("|");
    return `<div>· ${px}px, ${bl.replace("b"," bloco(s)")} → pico <b>${(+d.gb).toFixed(2)} GB</b>`
         + (d.onde ? ` <i>(${esc(d.onde)})</i>` : "") + `</div>`;
  });
  $("cfgVram").innerHTML =
    `<div>placa: <b>${VRAM.total_gb} GB</b> · base do motor: ~${VRAM.base_gb} GB`
  + ` · piso do desktop medido: ~1,6 GB</div>`
  + `<div style="margin-top:6px"><b>picos já medidos nesta máquina</b> (a previsão vem daqui):</div>`
  + (linhas.length ? linhas.join("") : "<div><i>nenhuma medição ainda — gere uma vez</i></div>");
  try{
    const c = await api().gsmde_caminhos();
    if(c && !c.__error__){
      // De onde cada peso VEM de fato — informacao, nao ajuste. Por isso vive
      // no balao de hint e nao ocupa uma secao da coluna, que e' onde moram as
      // coisas que se mexem.
      CAMINHOS_TXT = (c.itens || []).map(i=>
        `· ${i.nome}: ${i.caminho} [${i.disco}]`
        + (i.gb ? ` ${(+i.gb).toFixed(2)} GB` : "")).join(NL1);
    }
  }catch(_){}
}

/* ---------- prompt cresce quando os menus recolhem ---------- */
function toggleAll(){
  const abertos = Array.from(document.querySelectorAll("#groups > details"))
    .filter(d=> d.style.display !== "none" && d.open);
  const fechar = abertos.length > 0;
  document.querySelectorAll("#groups > details").forEach(d=> d.open = !fechar);
  $("toggleAll").textContent = fechar ? "⤢ abrir menus" : "⤢ recolher menus";
}

/* ================= boot ================= */
let booted = false;
async function boot(){
  if(booted) return;      // pywebviewready + fallback do timer nao podem ligar 2x
  booted = true;
  $("seed").disabled = $("seedRand").checked;
  $("seedRand").addEventListener("change", ()=> $("seed").disabled = $("seedRand").checked);
  $("auto").addEventListener("change", syncAuto); syncAuto();
  $("hires").addEventListener("change", syncHires); syncHires();
  $("ultra").addEventListener("change", syncUltra); syncUltra();
  $("backboneVar").addEventListener("change", syncBackboneVar); syncBackboneVar();
  montaGaleriaLoras();
  montaCor();
  montaI2i();
  montaAbas();
  $("cfgMedir").addEventListener("click", atualizaCfgVram);
  montaCivitai();
  montaAgenda();
  montaModelos();
  montaScanAdapt();
  montaCaminhos();
  montaRam();
  montaTE();
  $("largura").addEventListener("input", atualizaProp);
  $("altura").addEventListener("input", atualizaProp);
  $("hintBtn").addEventListener("click", ()=> setHint(!hintOn));
  setHint(true);
  atualizaProp(); pintaDets();

  $("genBtn").addEventListener("click", gerar);
  $("stopBtn").addEventListener("click", parar);
  $("verCentros").addEventListener("click", verCentros);
  $("reciclar").addEventListener("click", reciclar);
  $("toggleAll").addEventListener("click", toggleAll);
  $("logBtn").addEventListener("click", async ()=>{
    if($("logBox").style.display === "block"){
      $("logBox").style.display = "none";
      if($("viewImg").src) $("viewImg").style.display = "block";
      return;
    }
    const st = api() ? await api().gsmde_status() : null;
    $("viewImg").style.display = "none";
    showLog((st && st.log) || "(sem log ainda)");
  });
  // terminal em janela separada: fechar a janela NAO derruba o worker (ele vive
  // preso a janela principal do launcher), entao da p/ abrir e fechar a vontade.
  $("termBtn").addEventListener("click", async ()=>{
    if(!api()) return;
    const r = await api().gsmde_console();
    if(r && r.__error__) toast(r.__error__);
  });
  $("camAdd").addEventListener("click", ()=>{
    // camada nova = mais ganho, menos denoise, mais passos (a regra do pipeline)
    const ult = CAMADAS[CAMADAS.length-1] || {g:1.5, dn:0.45, st:30};
    CAMADAS.push({g:+((+ult.g || 1.5) + 0.35).toFixed(2),
                  dn:+Math.max(0.15, ult.dn-0.15).toFixed(2), st:ult.st+10, on:true});
    pintaCamadas();
  });
  $("camDel").addEventListener("click", ()=>{
    if(CAMADAS.length > 1){ CAMADAS.pop(); pintaCamadas(); }
  });
  $("infinita").addEventListener("change", ()=>{
    // sem seed aleatoria o loop repete a MESMA imagem — nao e' o que se quer garimpar
    if($("infinita").checked && !$("seedRand").checked){
      $("seedRand").checked = true;
      $("seedRand").dispatchEvent(new Event("change"));
      toast("seed aleatória ligada (senão o loop repete a mesma imagem)");
    }
  });
  $("centrosNone").addEventListener("click", ()=> marcaCentros("", ""));
  $("centrosDoAuto").addEventListener("click", async ()=>{
    const r = await verCentros();
    if(r && !r.__error__) marcaCentros(r.centros, r.globais);
  });
  $("detAdd").addEventListener("change", ()=>{
    const m = $("detAdd").value; if(!m) return;
    if(!DETS.some(d=>d.model===m)) DETS.push({model:m, prompt:"", dn:0.3, st:20});
    $("detAdd").value = ""; pintaDets();
  });

  if(!api()) return;
  const c = await api().gsmde_centros();
  CENTROS = (c && c.centros) || {};
  pintaCentros();
  // teto de centros = quantos estao INSTALADOS, nao um numero fixo. O limite de 10
  // era arbitrario; quem decide o custo e' a VRAM/tempo, e disso o painel avisa.
  // Sem teto artificial: o campo aceita qualquer numero. Antes era um slider
  // 1..10 e, com 425 centros instalados, a posicao do cursor nao dizia mais
  // nada — duas telas mostravam o cursor em lugares diferentes marcando 6.
  // Quem limita de verdade e' a VRAM, e disso o painel de custo ja avisa.
  // O rotulo vem do ROTEADOR, nao de CENTROS: gsmde_centros() le so' os
  // clusters c1/c2 e filtra pelos treinados, entao dizia "de 36" enquanto o
  // roteador escolhia entre 425. Mentia exatamente no numero que o usuario usa
  // p/ decidir o teto.
  try{
    const t = await api().gsmde_centros_total();
    if(t && t.ok){
      $("maxCentros_v").innerHTML =
        `de <b>${t.total}</b> disponíveis` +
        `<span style="opacity:.6"> (${t.treinados} treinados + ${t.externos} baixados)</span>`;
    }
  }catch(_){
    const nInst = Object.keys(CENTROS).length;
    if(nInst) $("maxCentros_v").textContent = `de ${nInst} conhecidos`;
  }
  const v = await api().gsmde_vram_tabela();
  if(v && !v.__error__) VRAM = v;
  const y = await api().gsmde_yolos();
  YOLOS = (y && y.yolos) || [];
  $("detAdd").innerHTML = `<option value="">+ adicionar alvo…</option>`
    + YOLOS.map(o=>`<option value="${esc(o.id)}">${esc(o.nome)}</option>`).join("");
  pintaDets();

  // restaura DEPOIS das listas: marcar um centro/tag exige que a caixa ja exista
  try{
    const prefs = await api().get_prefs();
    applyState(prefs && prefs.tools && prefs.tools[CHAVE_ESTADO]);
  }catch(_){}

  // RETOMA um job em andamento: o worker e' subprocesso e sobrevive a recarregar a
  // pagina, mas o polling vive no JS e morria junto — a geracao continuava no disco
  // com a UI dizendo "pronto", sem preview, sem imagem final e sem o loop infinito.
  try{
    const st0 = await api().gsmde_status();
    if(st0 && st0.running){
      etapasSeen = 0; pvSeen = -1;
      setBusy(true, st0.passo || "retomando…");
      clearInterval(pollTimer);
      pollTimer = setInterval(()=> poll($("infinita").checked, +$("seed").value || 0), 1200);
    }
  }catch(_){}
  // delegacao: camadas e alvos do detailer sao recriados a cada repintura, entao
  // ouvir no container pega tambem o que ainda nao existia.
  $("left").addEventListener("input", scheduleSave);
  $("left").addEventListener("change", scheduleSave);
  $("left").addEventListener("change", atualizaCusto);   // marcar centro muda a RAM
  // a conta de camadas depende dos passos; 'change' so' dispara ao sair do campo
  $("steps").addEventListener("input", atualizaCusto);
  atualizaCusto();
  $("groups").addEventListener("toggle", scheduleSave, true);
  window.addEventListener("beforeunload", ()=>{
    try{ api().save_tool_state(CHAVE_ESTADO, collectState()); }catch(_){}
  });
}

/* estado inicial do balao */
if(window.HELP && window.HELP._default){
  $("h-title").textContent = window.HELP._default.t;
  $("h-body").textContent  = window.HELP._default.b;
}


/* ---------------- alarme sonoro do overload ----------------
   Sintetizado com Web Audio (oscilador), sem arquivo nenhum: nada para baixar,
   nada para faltar na instalacao.

   Duas regras que evitam que isto vire um bug:
     1. dispara na TRANSICAO para o nivel, nao a cada medicao — ramMede() roda a
        cada tecla digitada no campo do buffer, e sem isso beeparia a cada letra;
     2. tem botao de silenciar. Alarme sem como desligar nao e' insistente, e'
        quebrado — e o usuario acabaria desligando a aba inteira.
   Enquanto o nivel 2 (ou o critico) persistir, ele repete no intervalo. E' o
   ponto: incomodar ate a configuracao ser arrumada. */
let AUDIO_CTX = null, ALARME_TIMER = null, ALARME_MUDO = false, ALARME_NIVEL = 0;

function bipe(freqs, dur, vol){
  try{
    AUDIO_CTX = AUDIO_CTX || new (window.AudioContext || window.webkitAudioContext)();
    if(AUDIO_CTX.state === "suspended") AUDIO_CTX.resume();   // politica de autoplay
    let t = AUDIO_CTX.currentTime;
    freqs.forEach(f=>{
      const osc = AUDIO_CTX.createOscillator(), g = AUDIO_CTX.createGain();
      osc.type = "square"; osc.frequency.value = f;
      g.gain.setValueAtTime(0, t);
      g.gain.linearRampToValueAtTime(vol, t + 0.01);
      g.gain.linearRampToValueAtTime(0, t + dur);
      osc.connect(g); g.connect(AUDIO_CTX.destination);
      osc.start(t); osc.stop(t + dur);
      t += dur + 0.04;
    });
  }catch(_){}
}

const ALARMES = {
  2: { freqs:[880, 660, 880], dur:0.14, vol:0.16, intervalo:9000 },
  3: { freqs:[1100, 700, 1100, 700], dur:0.17, vol:0.22, intervalo:5000 },
};

function alarmeAplica(nivel){
  if(nivel === ALARME_NIVEL) return;          // so' na transicao
  ALARME_NIVEL = nivel;
  clearInterval(ALARME_TIMER); ALARME_TIMER = null;
  const cfg = ALARMES[nivel];
  if(!cfg || ALARME_MUDO) return;
  bipe(cfg.freqs, cfg.dur, cfg.vol);
  ALARME_TIMER = setInterval(()=>{
    if(ALARME_MUDO || ALARME_NIVEL !== nivel){ clearInterval(ALARME_TIMER); return; }
    bipe(cfg.freqs, cfg.dur, cfg.vol);
  }, cfg.intervalo);
}

function balaoDevolve(){
  if(!hintOn){ $("help").style.display = "none"; document.body.classList.add("sem-hint"); }
}

function alarmeMudo(m){
  ALARME_MUDO = m;
  const b = $("ramMudo"); if(b){ b.textContent = m ? "🔇 mudo" : "🔊 som"; b.classList.toggle("on", !m); }
  if(m){ clearInterval(ALARME_TIMER); ALARME_TIMER = null; }
  else { const n = ALARME_NIVEL; ALARME_NIVEL = -1; alarmeAplica(n); }
}



/* ---------------- modo do text encoder ---------------- */
const TE_AVISOS = {
  cpu: "Não usa VRAM. Nesta máquina o bf16 na CPU é o caminho medido como melhor " +
       "(o torch ROCm vem sem MKL, e fp32 na CPU cai num caminho ingênuo).",
  sequencial: "O TE sobe para a placa só para codificar e desce antes do denoise. " +
       "As duas fases não competem no tempo, então pode caber mesmo em 8 GB — " +
       "o UNet ainda não está no lugar quando o TE sobe. Meça antes de confiar.",
  residente: "CLIP + CLIP-G somam ~1,4 GB à VRAM durante TODA a geração, por um " +
       "trabalho que dura segundos. Em 8 GB com o UNet em ~5,2 GB isto derrama."
};

async function montaTE(){
  if(!$("teOffload")) return;
  const a = api(); if(!a) return;
  const pinta = ()=>{ $("teAviso").textContent = TE_AVISOS[$("teOffload").value] || ""; };
  try{
    const r = await a.gsmde_te_config();
    if(r && r.ok){ $("teOffload").value = r.te_offload || "cpu";
                   $("teDtype").value = r.te_dtype || ""; }
  }catch(_){}
  pinta();
  const salva = async ()=>{
    pinta();
    try{ await a.gsmde_te_config($("teOffload").value, $("teDtype").value); }catch(_){}
    avisa("Modo do text encoder salvo — vale na próxima geração.");
  };
  if($("aotriton")){
    (async ()=>{ try{ const r = await a.gsmde_aotriton();
      if(r && r.ok) $("aotriton").checked = r.aotriton !== "0"; }catch(_){} })();
    $("aotriton").addEventListener("change", async ()=>{
      await a.gsmde_aotriton($("aotriton").checked);
      avisa("AOTRITON " + ($("aotriton").checked ? "ligado" : "desligado") +
            " — vale na próxima geração.");
    });
  }
  $("teOffload").addEventListener("change", salva);
  $("teDtype").addEventListener("change", salva);
}

/* ---------------- monitor ao vivo ----------------
   Enquanto gera, mostra RAM e VRAM MEDIDAS na barra de estagio. Existe para
   conferir a previsao contra o real: a aba de configuracao estima antes de
   comecar, e aqui se ve se a estimativa acertou. Se divergirem muito, e' a
   estimativa que esta errada — a mediana dos centros nao representa a selecao
   daquela geracao. */
let MON_T = 0;
async function monitorVivo(st){
  const el = $("monVivo"); if(!el) return;
  const agora = Date.now();
  if(agora - MON_T < 1500) return;      // o poll e' mais rapido que a medicao precisa
  MON_T = agora;
  try{
    const r = await api().ram_estado($("ramBuf") ? $("ramBuf").value : null,
                                     +$("maxCentros").value || 6);
    if(!r || !r.ok) return;
    const pct = Math.round(r.uso * 100);
    const cor = r.critico ? "#ff7b72" : (pct >= 80 ? "#ffa657" : "#7ee787");
    const vram = (st && st.vram_gb) ? ` · VRAM ${(+st.vram_gb).toFixed(1)} GB` : "";
    el.innerHTML = `<span style="color:${cor}">RAM ${pct}%</span>` +
      ` <span style="opacity:.55">(${r.livre_gb.toFixed(1)} GB livres)</span>${vram}`;
  }catch(_){}
}

/* ---------------- buffer de RAM e alertas de overload ----------------
   Tres regimes, e SO' UM interrompe:
     nivel 1  os centros passam do buffer escolhido       -> incomoda, gera
     nivel 2  passam 20% alem                             -> incomoda mais, gera
     CRITICO  a RAM do SISTEMA chega a 83%                -> bloqueia
   Os dois primeiros existem para ser chatos ate o usuario arrumar a
   configuracao; nenhum deles cancela nada. O terceiro nao e' sobre a escolha
   do usuario e sim sobre o estado da maquina — por isso e' o unico que para. */
let RAM_NIVEL = 0, RAM_DIAG = "";

async function ramMede(){
  const a = api(); if(!a || !$("ramSaida")) return null;
  let r;
  try { r = await a.ram_estado($("ramBuf").value, $("ramNc").value); }
  catch(e){ $("ramSaida").textContent = "erro: " + e; return null; }
  if(!r || !r.ok){ $("ramSaida").textContent = "erro: " + ((r && r.erro) || "?"); return null; }

  // PREVISAO: onde o sistema vai parar se estes centros forem carregados.
  // Calculado ANTES de RAM_NIVEL porque ele depende disto — deixar embaixo dava
  // ReferenceError (const em zona morta) e derrubava a medicao inteira.
  const fracCentros = r.total_gb > 0 ? r.preciso_gb / r.total_gb : 0;
  const previsto = r.uso + fracCentros;
  const impossivel = previsto > 1.0;              // nao cabe nem com 100% da RAM
  const pctPrev = Math.round(previsto * 100);

  RAM_NIVEL = (r.critico || impossivel) ? 3 : r.nivel;
  alarmeAplica(RAM_NIVEL);   // nivel 2 e critico tocam; 0 e 1 silenciam

  // A aba pisca para dizer ONDE arrumar — o usuario pode estar em qualquer
  // outra aba quando o problema aparece, e um alarme que nao aponta o caminho
  // so' gera irritacao sem conserto.
  const abaCfg = document.querySelector('.aba[data-aba="config"]');
  if(abaCfg) abaCfg.classList.toggle("piscando", RAM_NIVEL >= 2);

  if(RAM_NIVEL >= 2) showHelp("ramBuffer", true);
  else balaoDevolve();
  const pct = Math.round(r.uso * 100);
  const classe = r.critico ? "ram-crit" : (r.nivel === 2 ? "ram-n2"
                : (r.nivel === 1 ? "ram-n1" : "ram-ok"));
  // o quadradinho da legenda acompanha a cor da barra, senao a legenda diria
  // "verde = em uso" enquanto a barra esta laranja.
  const sufixo = r.critico ? "crit" : (r.nivel === 2 ? "n2" : (r.nivel === 1 ? "n1" : ""));
  const larguraPrev = Math.max(0, Math.min(100 - pct, Math.round(fracCentros * 100)));
  const agulha = Math.min(100, pctPrev);

  $("ramBarra").innerHTML =
    `<div class="ram-barra ${classe}${impossivel ? " impossivel" : ""}">
       <i class="usado" style="width:${pct}%"></i>
       <i class="previsto" style="left:${pct}%;width:${larguraPrev}%"></i>
       <span class="marca alerta" style="left:${Math.round(r.teto_alerta*100)}%"></span>
       <span class="marca crit" style="left:${Math.round(r.teto_critico*100)}%"></span>
       <span class="agulha" style="left:calc(${agulha}% - 1px)"></span>
     </div>
     <div class="mini" style="opacity:.6;margin-top:5px">
       agora ${pct}% · <b>previsto ${pctPrev}%</b> com os centros</div>
     <div class="ram-leg">
       <span><i class="sw usado ${sufixo}"></i>em uso agora (${pct}%)</span>
       <span><i class="sw prev"></i>o que os centros vão ocupar (${r.preciso_gb.toFixed(1)} GB)</span>
       <span><i class="sw livre"></i>sobra livre</span>
       <span><i class="sw ag"></i>onde a previsão termina</span>
       <span><i class="sw mk-al"></i>${Math.round(r.teto_alerta*100)}% — só avisa</span>
       <span><i class="sw mk-cr"></i>${Math.round(r.teto_critico*100)}% — bloqueia</span>
     </div>`;

  let tag = "";
  if(r.critico) tag = `<span class="ram-tag crit">⛔ CRÍTICO — geração bloqueada</span>`;
  else if(r.nivel === 2) tag = `<span class="ram-tag n2">⚠️ OVERLOAD nível 2</span>`;
  else if(r.nivel === 1) tag = `<span class="ram-tag n1">⚠️ OVERLOAD nível 1</span>`;

  const linhas = [
    `${r.n_centros} centros × ${r.mediana_centro_gb.toFixed(2)} GB = <b>${r.preciso_gb.toFixed(2)} GB</b>`,
    `buffer: <b>${(+r.buffer_gb).toFixed(1)} GB</b> · RAM livre ${r.livre_gb.toFixed(1)} de ${r.total_gb.toFixed(1)} GB`,
  ];
  if(r.nivel) linhas.push(`estouro de <b>${Math.round(r.excesso*100)}%</b> sobre o buffer` +
    (r.nivel === 2 ? " — reduza os centros ou aumente o buffer" : " — cabe apertado"));
  if(r.critico) linhas.push("A máquina está sem RAM. Feche algo: gerar agora jogaria o sistema em swap.");
  if(impossivel){
    const sobra = (r.total_gb * (1 - r.uso)).toFixed(1);
    linhas.push(`<span class="ram-impossivel">⛔ NÃO CABE FISICAMENTE:</span> ` +
      `${r.n_centros} centros pedem ${r.preciso_gb.toFixed(1)} GB e a máquina tem ` +
      `${sobra} GB livres de ${r.total_gb.toFixed(1)} GB no total. ` +
      `Nem usando 100% da RAM daria. ` +
      // A sugestao mira o teto de BLOQUEIO, nao os 100%: caber "exatamente na
      // RAM toda" e' tao inutil quanto nao caber — pararia no limite de 83%.
      `Para ficar abaixo do bloqueio de ${Math.round(r.teto_critico*100)}%, ` +
      `use no maximo ~${Math.max(1, Math.floor(
         (r.total_gb * (r.teto_critico - r.uso)) / r.mediana_centro_gb))} centros.`);
  }
  // texto que o balao mostra: o QUE, o PORQUE e o COMO arrumar, nessa ordem.
  RAM_DIAG = r.critico
    ? `A maquina esta com ${Math.round(r.uso*100)}% da RAM em uso (bloqueio em `
      + `${Math.round(r.teto_critico*100)}%). A geracao esta BLOQUEADA. `
      + `Feche algum programa aberto — nao e' a configuracao do GSMDE, e o estado da maquina.`
    : (r.nivel >= 2
      ? `Voce pediu ${r.n_centros} centros, que ocupam ~${r.preciso_gb.toFixed(1)} GB, `
        + `mas o buffer esta em ${(+r.buffer_gb).toFixed(1)} GB — ${Math.round(r.excesso*100)}% acima. `
        + `Arrume em Configuracao > Buffer de RAM: aumente o buffer OU reduza `
        + `"Max. centros" na aba Gerar. A geracao continua funcionando; o alarme `
        + `so' para quando os numeros fecharem.`
      : "");
  $("ramSaida").innerHTML = (tag ? tag + "<br>" : "") + linhas.join("<br>");
  return r;
}

function montaRam(){
  if(!$("ramBuf")) return;
  ["ramBuf","ramNc"].forEach(id=> $(id).addEventListener("input", ramMede));
  $("ramMedir").addEventListener("click", ramMede);
  if($("ramMudo")) $("ramMudo").addEventListener("click", ()=> alarmeMudo(!ALARME_MUDO));
  alarmeMudo(false);
  $("ramSalvar").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    const r = await a.ram_salvar_buffer($("ramBuf").value);
    if(r && r.ok){ avisa(""); ramMede(); } else avisa("nao salvou: " + ((r&&r.erro)||"?"));
  });
  // o campo de centros da aba Gerar alimenta a simulacao: mexer la' reflete aqui
  const mc = $("maxCentros");
  if(mc) mc.addEventListener("input", ()=>{ $("ramNc").value = mc.value; ramMede(); });
  ramMede();
}

/* ---------------- pastas (fonte unica dos caminhos) ---------------- */
const CM = { estado: {} };

async function cmCarrega(){
  const a = api(); if(!a || !$("cmLista")) return;
  let r;
  try { r = await a.caminhos_ler(); } catch(e){ $("cmAviso").textContent = "erro: " + e; return; }
  if(!r || !r.ok){ $("cmAviso").textContent = "erro: " + ((r && r.erro) || "?"); return; }
  CM.estado = r.caminhos || {};
  const faltando = r.faltando || [];
  $("cmAviso").innerHTML = faltando.length
    ? `⚠️ não existe(m): <b>${faltando.join(", ")}</b>`
    : "✅ todos os caminhos existem";

  $("cmLista").innerHTML = Object.entries(CM.estado).map(([nome, v])=>{
    const ok = v.existe ? "" : ' style="color:#ff7b72"';
    return `<div style="margin:8px 0">
      <label class="mini"${ok}>${nome} <span style="opacity:.55">— ${v.papel}</span></label>
      <div style="display:flex;gap:4px">
        <input class="cmIn" data-nome="${nome}" data-tipo="${v.tipo}" type="text"
               value="${v.caminho.replace(/"/g,'&quot;')}" style="flex:1" />
        <button class="mini-btn cmPick" data-nome="${nome}" data-tipo="${v.tipo}">…</button>
      </div>
      <div class="mini" style="opacity:.5">${v.disco}: · ${v.origem}${v.existe ? "" : " · NÃO EXISTE"}</div>
    </div>`;
  }).join("");

  $("cmLista").querySelectorAll(".cmPick").forEach(b=>
    b.addEventListener("click", async ()=>{
      const r2 = await a.caminhos_escolher(b.dataset.nome, b.dataset.tipo);
      if(r2 && r2.ok){
        const inp = $("cmLista").querySelector(`.cmIn[data-nome="${r2.nome}"]`);
        if(inp) inp.value = r2.caminho;
      }
    }));
}

function montaCaminhos(){
  if(!$("cmSalvar")) return;
  cmCarrega();

  $("cmSalvar").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    const novos = {};
    $("cmLista").querySelectorAll(".cmIn").forEach(i=>{
      const orig = (CM.estado[i.dataset.nome] || {});
      // so' manda o que o usuario REALMENTE mudou; reenviar o valor detectado
      // viraria override e congelaria a deteccao automatica sem ele querer.
      if(i.value.trim() !== (orig.caminho || "")) novos[i.dataset.nome] = i.value.trim();
    });
    if(!Object.keys(novos).length){ avisa("Nada mudou."); return; }
    const r = await a.caminhos_salvar(novos);
    if(r && r.ok){ avisa("Pastas salvas — valem na próxima geração."); cmCarrega(); }
    else avisa("Não salvou: " + ((r && r.erro) || "?"));
  });

  $("cmDetectar").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    $("cmDetectado").textContent = "procurando nos discos…";
    const r = await a.caminhos_detectar();
    const rs = (r && r.raizes) || [];
    $("cmDetectado").innerHTML = rs.length
      ? "bibliotecas encontradas:<br>" + rs.map(x=>
          `· <b>${x.caminho}</b> — ${x.marcas} marcas, ${x.gb} GB
           <button class="mini-btn cmUsar" data-p="${x.caminho}">usar</button>`).join("<br>")
      : "<i>nenhuma pasta com cara de biblioteca de modelos</i>";
    $("cmDetectado").querySelectorAll(".cmUsar").forEach(b=>
      b.addEventListener("click", ()=>{
        const inp = $("cmLista").querySelector('.cmIn[data-nome="modelos_raiz"]');
        if(inp) inp.value = b.dataset.p;
        avisa("Raiz preenchida — clique em salvar para valer.");
      }));
  });
}


/* ---------------- scan & adapt ----------------
   O GSMDE nao publica "modelos GSMDE": ele adota LoRAs SDXL de qualquer origem.
   O que falta em um LoRA baixado nao e' compatibilidade de peso — e' saber QUE
   DOMINIO ele cobre, senao o roteador nunca o escolhe. Isto le esse dominio do
   proprio arquivo (ss_tag_frequency do kohya) e mostra antes de instalar. */
let SA_ULTIMO = null;

function montaScanAdapt(){
  if(!$("saScan")) return;
  $("saPick").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    const r = await a.caminhos_escolher("lora", "arquivo");
    if(r && r.ok) $("saCaminho").value = r.caminho;
  });

  $("saScan").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    const p = $("saCaminho").value.trim();
    if(!p){ avisa("Informe o caminho do .safetensors."); return; }
    $("saSaida").textContent = "lendo metadados…";
    $("saInstalar").style.display = "none";
    const r = await a.centros_adaptar(p);
    if(!r || !r.ok){ $("saSaida").textContent = "erro: " + ((r && r.erro) || "?"); return; }
    SA_ULTIMO = r;
    const linhas = [
      `<b>${esc(r.titulo || r.arquivo)}</b> · ${r.gb} GB` +
        (r.rank ? ` · rank ${r.rank}` : "") +
        (r.n_imagens_treino ? ` · ${r.n_imagens_treino} imgs de treino` : ""),
      `fonte do domínio: <b>${r.fonte_dominio}</b>` +
        (r.base_treino ? ` · treinado sobre <i>${esc(r.base_treino)}</i>` : "")
    ];
    if(r.adaptavel){
      linhas.push("domínio detectado (o que o roteador vai usar):");
      linhas.push("<div style='opacity:.8'>" + r.dominio.slice(0,8).map(d=>
        `· <b>${esc(d.tag)}</b> <span style="opacity:.6">(peso ${d.peso})</span>`).join("<br>") + "</div>");
      $("saInstalar").style.display = "";
    } else {
      linhas.push(`⚠️ ${r.motivo}`);
    }
    $("saSaida").innerHTML = linhas.join("<br>");
  });

  $("saInstalar").addEventListener("click", async ()=>{
    const a = api(); if(!a || !SA_ULTIMO) return;
    $("saSaida").innerHTML += "<br>instalando…";
    const r = await a.centros_instalar($("saCaminho").value.trim(), "local");
    $("saSaida").innerHTML += r && r.ok
      ? `<br>✅ instalado — o cluster agora tem <b>${r.n}</b> centro(s)`
      : `<br>erro: ${(r && r.erro) || "?"}`;
  });
}

/* ---------------- modelos: busca nas duas lojas ---------------- */
async function mdStatus(){
  const a = api(); if(!a || !$("hfStatus")) return;
  try {
    const s = await a.hf_status();
    $("hfStatus").textContent = s && s.logado
      ? `✅ conectado${s.usuario ? " como " + s.usuario : ""} (${s.oauth ? "OAuth" : "token"})`
      : "não conectado";
    if(s && s.client_id && !$("hfClientId").value) $("hfClientId").value = s.client_id;
    $("hfLogout").style.display = (s && s.logado) ? "" : "none";
  } catch(e){ $("hfStatus").textContent = "erro: " + e; }
}

function mdCard(it){
  if(it.erro) return `<div style="opacity:.6">⚠️ ${it.fonte}: ${it.erro}</div>`;
  const loja = it.fonte === "hf" ? "🤗" : "🅲";
  const dl = it.downloads ? ` · ${(+it.downloads).toLocaleString()} ↓` : "";
  const gb = it.gb ? ` · ${it.gb} GB` : "";
  const trig = (it.trigger && it.trigger.length)
    ? `<br><span style="opacity:.6">trigger: ${it.trigger.slice(0,3).join(", ")}</span>` : "";
  // a etiqueta de login e' o pedido explicito: sem ela o download volta HTML
  // de erro em vez do peso, e so' se descobre depois de baixar.
  const trava = it.precisa_login
    ? ` <span style="color:#ffa657" title="precisa estar logado">🔒 exige login</span>` : "";
  return `<div style="padding:6px 0;border-bottom:1px solid rgba(255,255,255,.07)">
    ${loja} <b>${it.nome}</b>${trava}<br>
    <span style="opacity:.6">${it.base || ""}${dl}${gb}</span>${trig}
    <button class="mini-btn mdBaixar" data-fonte="${it.fonte}"
      data-id="${it.id}" data-url="${it.url || ""}">baixar como centro</button>
  </div>`;
}

function montaModelos(){
  if(!$("mdBuscar")) return;

  $("mdBuscar").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    const termo = $("mdBusca").value.trim();
    if(!termo){ avisa("Digite algo para buscar."); return; }
    $("mdResultados").innerHTML = "buscando…";
    const r = await a.centros_buscar(termo, $("mdFonte").value, 20);
    const itens = (r && r.itens) || [];
    $("mdResultados").innerHTML = itens.length
      ? itens.map(mdCard).join("") : "<i>nada encontrado</i>";
    $("mdResultados").querySelectorAll(".mdBaixar").forEach(b=>
      b.addEventListener("click", ()=> avisa(
        "O download ainda não está ligado nesta build — a busca e o catálogo já estão. " +
        "Ver a nota sobre o que falta.")));
  });

  $("hfLogin").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    const cid = $("hfClientId").value.trim();
    if(!cid){ avisa("Informe o client_id do app da HF."); return; }
    const b = $("hfLogin"), t = b.textContent;
    b.disabled = true; b.textContent = "⏳ aguardando o navegador…";
    const r = await a.hf_login(cid);
    if(!(r && r.ok)) avisa("Login não concluído: " + ((r && r.erro) || "?"));
    b.disabled = false; b.textContent = t; mdStatus();
  });

  $("hfSalvarToken").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    const r = await a.hf_set_token($("hfToken").value.trim());
    $("hfToken").value = "";
    if(!(r && r.ok)) avisa("Token rejeitado: " + ((r && r.erro) || "?"));
    mdStatus();
  });
  $("hfLogout").addEventListener("click", async ()=>{
    const a = api(); if(!a) return; await a.hf_logout(); mdStatus();
  });

  $("mdExtrair").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    const p = $("mdUnetEnt").value.trim();
    if(!p){ avisa("Informe o caminho do checkpoint."); return; }
    $("mdUnetSaida").textContent = "extraindo… (pode levar ~20s)";
    const r = await a.hf_extrair_unet(p);
    if(!r || !r.ok){ $("mdUnetSaida").textContent = "erro: " + ((r && r.erro) || "?"); return; }
    $("mdUnetSaida").innerHTML =
      `${r.compativel ? "✅ compatível" : "⚠️ " + r.aviso}<br>` +
      `família ${r.forma.familia} · contexto ${r.forma.ctx_cross_attn.join("/")} · ` +
      `${r.params_bi} B params · ${r.gb} GB<br>` +
      `<span style="opacity:.6">${r.caminho}</span>`;
  });

  $("mdComparar").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    const p = $("mdUnetEnt").value.trim();
    if(!p){ avisa("Informe o caminho."); return; }
    $("mdUnetSaida").textContent = "comparando…";
    const r = await a.hf_comparar(p);
    $("mdUnetSaida").innerHTML = (r && r.ok)
      ? `divergência <b>${r.divergencia_media}</b> — ${r.veredito}<br>` +
        `<span style="opacity:.6">${r.chaves_comuns} chaves em comum, ${r.amostradas} amostradas</span>`
      : "erro: " + ((r && r.erro) || "?");
  });

  $("mdReindexar").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    $("mdRankSaida").textContent = "indexando…";
    const r = await a.centros_reindexar("todas");
    $("mdRankSaida").innerHTML = (r && r.ok)
      ? r.resultados.map(x=> x.erro ? `⚠️ ${x.erro}`
          : `${x.origem}: ${x.n} centro(s), ${x.com_dominio||0} com domínio`).join("<br>")
      : "erro: " + ((r && r.erro) || "?");
  });

  $("mdRanking").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    $("mdRankSaida").textContent = "lendo…";
    const r = await a.centros_ranking(2.0);
    if(!r || !r.ok){ $("mdRankSaida").textContent = "erro: " + ((r && r.erro) || "?"); return; }
    const lin = (r.ranking || []).slice(0, 12).map((x, i)=>{
      const quente = (r.preload || []).includes(x.nome) ? " 🔥" : "";
      return `${String(i+1).padStart(2)}. ${x.nome} — ${x.pontos} pts (${x.n} usos)${quente}`;
    });
    $("mdRankSaida").innerHTML = (lin.length ? lin.join("<br>") : "<i>sem uso registrado</i>") +
      `<br><br>pré-carregar: <b>${(r.preload||[]).length}</b> centro(s), ${r.gb} de ${r.orcamento_gb} GB`;
  });
}

/* ---------------- agenda de posts ----------------
   Calendario do mes com o que ja saiu (API oficial) e o que esta agendado
   (sessao). Clicar num dia abre o painel de agendamento daquele dia. */
const AG = { ano: 0, mes: 0, dias: {}, sel: null };
const MESES = ["janeiro","fevereiro","março","abril","maio","junho","julho",
               "agosto","setembro","outubro","novembro","dezembro"];

async function agCarrega(){
  const a = api(); if(!a) return;
  if(!AG.ano){ const h = new Date(); AG.ano = h.getFullYear(); AG.mes = h.getMonth()+1; }
  $("agCal").innerHTML = '<div class="mini">carregando…</div>';
  let r;
  try { r = await a.civitai_agenda_mes(AG.ano, AG.mes); }
  catch(e){ $("agCal").innerHTML = ""; $("agAvisos").textContent = "falha: " + e; return; }
  if(!r || !r.ok){ $("agCal").innerHTML = "";
    $("agAvisos").textContent = "erro: " + ((r && r.erro) || "?"); return; }
  AG.dias = r.dias || {};
  $("agMes").textContent = `${MESES[AG.mes-1]} de ${AG.ano}`;
  $("agAvisos").innerHTML = (r.avisos || []).map(t=>`⚠️ ${t}`).join("<br>");
  agDesenha(r.malha);
}

function agDesenha(malha){
  const dom = ["D","S","T","Q","Q","S","S"];
  let h = '<table class="agtab"><thead><tr>' +
          dom.map(d=>`<th>${d}</th>`).join("") + '</tr></thead><tbody>';
  for(const semana of malha.semanas){
    h += "<tr>";
    for(const dia of semana){
      if(!dia){ h += '<td class="vazio"></td>'; continue; }
      const itens = AG.dias[dia] || [];
      const nAg = itens.filter(i=>i.estado === "agendado").length;
      const nPub = itens.length - nAg;
      const hoje = dia === malha.hoje ? " hoje" : "";
      const num = +dia.slice(-2);
      let pts = "";
      if(nPub) pts += `<span class="pt pub" title="${nPub} publicado(s)">●</span>`;
      if(nAg)  pts += `<span class="pt agd" title="${nAg} agendado(s)">◆</span>`;
      h += `<td class="dia${hoje}" data-dia="${dia}"><div class="n">${num}</div>${pts}</td>`;
    }
    h += "</tr>";
  }
  $("agCal").innerHTML = h + "</tbody></table>";
  $("agCal").querySelectorAll("td.dia").forEach(td=>
    td.addEventListener("click", ()=> agAbreDia(td.dataset.dia)));
}

function agAbreDia(dia){
  AG.sel = dia;
  $("agCal").querySelectorAll("td.dia").forEach(td=>
    td.classList.toggle("sel", td.dataset.dia === dia));
  $("agDia").style.display = "";
  $("agDiaTitulo").textContent = "📅 " + dia;
  const itens = AG.dias[dia] || [];
  $("agDiaItens").innerHTML = itens.length
    ? itens.map(i=>{
        const q = i.quando ? new Date(i.quando).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"}) : "—";
        const tag = i.estado === "agendado" ? "◆ agendado" : "● publicado";
        return `<div>${tag} · ${q} · post <b>${i.post_id}</b> · ${i.imagens} img` +
               (i.titulo ? ` · ${i.titulo}` : "") + "</div>";
      }).join("")
    : "<i>nada neste dia</i>";
}

function montaAgenda(){
  if(!$("agCal")) return;
  const passo = (d)=>{
    let m = AG.mes + d, y = AG.ano;
    if(m < 1){ m = 12; y--; } if(m > 12){ m = 1; y++; }
    AG.mes = m; AG.ano = y; agCarrega();
  };
  $("agAnt").addEventListener("click", ()=> passo(-1));
  $("agProx").addEventListener("click", ()=> passo(1));
  $("agRecarregar").addEventListener("click", agCarrega);

  // Login SEMPRE no navegador padrao. A versao anterior abria um webview
  // embutido em civitai.com/login, que redirige para o Google — ou seja, pedia
  // a senha dentro de uma janela do proprio app, fora do alcance do
  // gerenciador de senhas. Removido.
  $("agSessao").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    const st = await a.civitai_status();
    if(!st || !st.client_id){
      avisa("Configure o client_id do CivitAI na aba Configuração primeiro.");
      return;
    }
    const b = $("agSessao"), t = b.textContent;
    b.disabled = true; b.textContent = "⏳ aguardando o navegador…";
    const r = await a.civitai_login();
    b.disabled = false; b.textContent = t;
    if(r && r.ok){ avisa(""); agCarrega(); }
    else avisa("Login não concluído: " + ((r && r.erro) || "?"));
  });

  $("agSalvar").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    if(!AG.sel){ avisa("Escolha um dia no calendário."); return; }
    const pid = parseInt($("agPostId").value, 10);
    if(!pid){ avisa("Informe o id do post no CivitAI."); return; }
    const quando = `${AG.sel}T${$("agHora").value || "12:00"}:00`;
    if(new Date(quando) < new Date() &&
       !confirm("Essa data já passou — o post sairia na hora. Continuar?")) return;
    const r = await a.civitai_agendar(pid, quando);
    if(r && r.ok){ avisa(""); agCarrega(); }
    else avisa("Não agendou: " + ((r && (r.erro || r.resposta)) || "?"));
  });
}

/* ---------------- conta CivitAI (login opcional) ----------------
   O token vive no Python (cifrado com DPAPI); daqui so' se ve o estado.
   O login BLOQUEIA ate o usuario concluir no navegador, entao o botao
   avisa e se desabilita — senao parece travado. */
async function cvPinta(st){
  const el = $("cvStatus"); if(!el) return;
  if(!st || st.erro){ el.textContent = st && st.erro ? "erro: " + st.erro : "—"; return; }
  if(st.logado){
    const via = st.oauth ? "OAuth" : "API key";
    const quem = st.usuario ? ` como <b>${st.usuario}</b>` : "";
    let val = "";
    if(st.oauth && st.expira_em){
      const min = Math.round((st.expira_em*1000 - Date.now())/60000);
      val = min > 0 ? ` · token renova em ${min}min` : " · renovando…";
    }
    el.innerHTML = `✅ conectado via ${via}${quem}${val}` +
      (st.escopos ? `<br><span style="opacity:.6">escopos: ${st.escopos}</span>` : "");
  } else {
    el.textContent = "não conectado";
  }
  if(st.client_id && !$("cvClientId").value) $("cvClientId").value = st.client_id;
  $("cvLogout").style.display = st.logado ? "" : "none";
}

async function cvStatus(){
  const a = api(); if(!a) return;
  try { cvPinta(await a.civitai_status()); } catch(e){ cvPinta({erro:String(e)}); }
}

function montaCivitai(){
  if(!$("cvLogin")) return;
  cvStatus();

  $("cvLogin").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    const cid = $("cvClientId").value.trim();
    if(!cid){ avisa("Informe o client_id de um app registrado em civitai.com."); return; }
    const b = $("cvLogin"), txt = b.textContent;
    b.disabled = true; b.textContent = "⏳ aguardando o navegador…";
    $("cvStatus").textContent = "Abri a página de login no seu navegador padrão. " +
      "Conclua por lá e volte — esta janela espera até 3 minutos.";
    try {
      const r = await a.civitai_login(cid);
      if(r && r.ok){ avisa(""); await cvStatus(); }
      else avisa("Login não concluído: " + ((r && r.erro) || "motivo desconhecido"));
    } catch(e){ avisa("Falha no login: " + e); }
    b.disabled = false; b.textContent = txt;
    cvStatus();
  });

  $("cvSalvarKey").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    const k = $("cvApiKey").value.trim();
    if(!k){ avisa("Cole a API key primeiro."); return; }
    const r = await a.civitai_set_api_key(k);
    $("cvApiKey").value = "";                       // nao deixa a chave na tela
    if(r && r.ok) avisa(""); else avisa("Chave rejeitada: " + ((r && r.erro) || "?"));
    cvStatus();
  });

  $("cvLogout").addEventListener("click", async ()=>{
    const a = api(); if(!a) return;
    await a.civitai_logout(); cvStatus();
  });
}

window.addEventListener("pywebviewready", boot);
if(window.pywebview && window.pywebview.api) boot();
else setTimeout(boot, 2000);   // fora do pywebview a UI ainda monta (so nao gera)
