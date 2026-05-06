/* global React, ReactDOM, TweaksPanel, useTweaks, TweakSection, TweakSlider, TweakColor, TweakRadio, TweakToggle, TweakSelect */

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "bg": "#f4ebd9",
  "ink": "#2b2419",
  "accentRed": "#b6432a",
  "accentMustard": "#c98e2b",
  "accentGreen": "#3f5d3a",
  "accentBlue": "#2f5b75",
  "displayFont": "Fraunces",
  "bodyFont": "Lora",
  "monoFont": "JetBrains Mono",
  "displayScale": 1.0,
  "bodyScale": 1.0,
  "showSlideNum": true,
  "showSlideTag": true,
  "showFootnote": true,
  "animations": true,
  "animationSpeed": 1.0,
  "patternOpacity": 0.12,
  "shadowStrength": 1.0,
  "darkMode": false
}/*EDITMODE-END*/;

const DISPLAY_FONTS = ['Fraunces', 'Playfair Display', 'DM Serif Display', 'Lora'];
const BODY_FONTS = ['Lora', 'Noto Sans TC', 'Source Serif 4', 'IBM Plex Sans'];
const MONO_FONTS = ['JetBrains Mono', 'IBM Plex Mono', 'Fira Code', 'Courier New'];

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);

  React.useEffect(() => {
    const root = document.documentElement;

    // colors
    if (t.darkMode) {
      root.style.setProperty('--bg', '#1a1610');
      root.style.setProperty('--bg-soft', '#2a2218');
      root.style.setProperty('--paper', '#241d14');
      root.style.setProperty('--ink', '#f4ebd9');
      root.style.setProperty('--ink-soft', '#c9bca0');
      root.style.setProperty('--ink-faint', '#8b7e68');
      root.style.setProperty('--line', '#5a4f3e');
      root.style.setProperty('--shadow', '6px 6px 0 #0a0805');
    } else {
      root.style.setProperty('--bg', t.bg);
      root.style.setProperty('--bg-soft', '#ebe0c8');
      root.style.setProperty('--paper', '#faf3e3');
      root.style.setProperty('--ink', t.ink);
      root.style.setProperty('--ink-soft', '#5a4f3e');
      root.style.setProperty('--ink-faint', '#8b7e68');
      root.style.setProperty('--line', '#c8b896');
      root.style.setProperty('--shadow', `${6*t.shadowStrength}px ${6*t.shadowStrength}px 0 ${t.ink}`);
    }
    root.style.setProperty('--accent-red', t.accentRed);
    root.style.setProperty('--accent-mustard', t.accentMustard);
    root.style.setProperty('--accent-green', t.accentGreen);
    root.style.setProperty('--accent-blue', t.accentBlue);

    // fonts — inject CSS rule
    let s = document.getElementById('__tweak_font_style');
    if (!s) {
      s = document.createElement('style');
      s.id = '__tweak_font_style';
      document.head.appendChild(s);
    }
    s.textContent = `
      .h-disp, .h-1, .h-2, .h-3, .stamp, .card-label, .tagline, .slide-tag {
        font-family: '${t.displayFont}', serif !important;
      }
      .body-l, .body-m {
        font-family: '${t.bodyFont}', 'Noto Sans TC', serif !important;
      }
      .mono, .eq-big, code, pre {
        font-family: '${t.monoFont}', monospace !important;
      }
      .h-disp { font-size: ${130 * t.displayScale}px !important; }
      .h-1 { font-size: ${84 * t.displayScale}px !important; }
      .h-2 { font-size: ${56 * t.displayScale}px !important; }
      .body-l { font-size: ${32 * t.bodyScale}px !important; }
      .body-m { font-size: ${26 * t.bodyScale}px !important; }
      .slide-num { display: ${t.showSlideNum ? 'block' : 'none'} !important; }
      .slide-tag { display: ${t.showSlideTag ? 'inline-block' : 'none'} !important; }
      .slide-foot { display: ${t.showFootnote ? 'block' : 'none'} !important; }
      .dot-pattern { opacity: ${t.patternOpacity / 0.12} !important; }
      ${!t.animations ? `
        deck-stage > section * { animation: none !important; transform: none !important; opacity: 1 !important; }
      ` : `
        deck-stage > section[data-deck-active] * {
          animation-duration: ${0.8 / t.animationSpeed}s !important;
        }
      `}
    `;
  }, [t]);

  return (
    <TweaksPanel title="Tweaks">
      <TweakSection title="🎨 Theme">
        <TweakToggle label="Dark mode" value={t.darkMode} onChange={v => setTweak('darkMode', v)} />
        <TweakColor
          label="Background"
          value={t.bg}
          options={['#f4ebd9', '#f5f0e4', '#ebe5d3', '#f8f3e8', '#e8dfc6']}
          onChange={v => setTweak('bg', v)}
        />
        <TweakColor
          label="Ink"
          value={t.ink}
          options={['#2b2419', '#1a1410', '#3a2e1f', '#2d2a25']}
          onChange={v => setTweak('ink', v)}
        />
      </TweakSection>

      <TweakSection title="🖌 Accent palette">
        <TweakColor
          label="Red"
          value={t.accentRed}
          options={['#b6432a', '#c14d2e', '#a23a2a', '#8e3a25', '#d44a2a']}
          onChange={v => setTweak('accentRed', v)}
        />
        <TweakColor
          label="Mustard"
          value={t.accentMustard}
          options={['#c98e2b', '#d4a02e', '#b87a26', '#e0a830']}
          onChange={v => setTweak('accentMustard', v)}
        />
        <TweakColor
          label="Green"
          value={t.accentGreen}
          options={['#3f5d3a', '#4a6b3e', '#355030', '#5a7a4e']}
          onChange={v => setTweak('accentGreen', v)}
        />
        <TweakColor
          label="Blue"
          value={t.accentBlue}
          options={['#2f5b75', '#3a6b8a', '#264c63', '#456f8a']}
          onChange={v => setTweak('accentBlue', v)}
        />
      </TweakSection>

      <TweakSection title="✒ Typography">
        <TweakSelect label="Display font" value={t.displayFont} options={DISPLAY_FONTS} onChange={v => setTweak('displayFont', v)} />
        <TweakSelect label="Body font" value={t.bodyFont} options={BODY_FONTS} onChange={v => setTweak('bodyFont', v)} />
        <TweakSelect label="Mono font" value={t.monoFont} options={MONO_FONTS} onChange={v => setTweak('monoFont', v)} />
        <TweakSlider label="Display scale" value={t.displayScale} min={0.7} max={1.4} step={0.05} onChange={v => setTweak('displayScale', v)} />
        <TweakSlider label="Body scale" value={t.bodyScale} min={0.8} max={1.3} step={0.05} onChange={v => setTweak('bodyScale', v)} />
      </TweakSection>

      <TweakSection title="🎬 Motion">
        <TweakToggle label="Animations on" value={t.animations} onChange={v => setTweak('animations', v)} />
        <TweakSlider label="Speed" value={t.animationSpeed} min={0.5} max={2.5} step={0.1} onChange={v => setTweak('animationSpeed', v)} />
      </TweakSection>

      <TweakSection title="🧱 Chrome">
        <TweakToggle label="Slide tag (top-left)" value={t.showSlideTag} onChange={v => setTweak('showSlideTag', v)} />
        <TweakToggle label="Slide number" value={t.showSlideNum} onChange={v => setTweak('showSlideNum', v)} />
        <TweakToggle label="Foot italic" value={t.showFootnote} onChange={v => setTweak('showFootnote', v)} />
        <TweakSlider label="Pattern dots" value={t.patternOpacity} min={0} max={0.3} step={0.01} onChange={v => setTweak('patternOpacity', v)} />
        <TweakSlider label="Shadow strength" value={t.shadowStrength} min={0} max={2} step={0.1} onChange={v => setTweak('shadowStrength', v)} />
      </TweakSection>
    </TweaksPanel>
  );
}

const root = document.createElement('div');
document.body.appendChild(root);
ReactDOM.createRoot(root).render(<App />);
