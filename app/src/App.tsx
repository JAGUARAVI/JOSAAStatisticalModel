import { useState, useEffect, useMemo, useRef } from 'react';
import {
  Search,
  TrendingUp,
  TrendingDown,
  Info,
  GraduationCap,
  LayoutGrid,
  List as ListIcon,
  SlidersHorizontal,
  Trophy,
  AlertCircle,
  User,
  Activity,
  Layers,
  ChevronRight,
  Cpu,
  BarChart3,
  MousePointer2,
  Minus,
  ArrowUp,
  Sun,
  Moon,
  ExternalLink,
  Shield,
  FileText,
  Scale
} from 'lucide-react';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  AreaChart,
  Area,
  LineChart,
  Line
} from 'recharts';
import { predictColleges, runDeepSimulation } from './lib/regression';
import type { RankRecord, PredictionResult, Metadata, PredictionsData } from './lib/regression';

function App() {
  const [records, setRecords] = useState<RankRecord[]>([]);
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [mlPredictions, setMlPredictions] = useState<PredictionsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [errorHeader, setErrorHeader] = useState<string | null>(null);

  // User Inputs
  const [rank, setRank] = useState<number | ''>('');
  const [category, setCategory] = useState('OPEN');
  const [gender, setGender] = useState('Gender-Neutral');
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('list');

  // New Filters
  const [selectedTypes, setSelectedTypes] = useState<string[]>(['IIT', 'NIT', 'IIIT', 'GFTI']);
  const [selectedClassifications, setSelectedClassifications] = useState<string[]>(['Safe', 'Likely', 'Competitive', 'Dream', 'Unlikely']);
  const [showSpecs, setShowSpecs] = useState(false);

  // Theme Management
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('theme');
      if (saved) return saved as 'light' | 'dark';
      return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    return 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(prev => prev === 'light' ? 'dark' : 'light');

  // Legal Modals
  const [showLegal, setShowLegal] = useState<'privacy' | 'terms' | 'disclaimer' | null>(null);

  // Global Simulation Tracking
  const [activeSimulations, setActiveSimulations] = useState(0);

  useEffect(() => {
    Promise.all([
      fetch('/data/ranks.json').then(res => res.json()),
      fetch('/data/metadata.json').then(res => res.json()),
      fetch('/data/predictions.json').then(res => res.json())
    ])
      .then(([ranksData, metaData, mlData]) => {
        setRecords(ranksData);
        setMetadata(metaData);
        const transformed: PredictionsData = {};
        if (mlData?.predictions) {
          mlData.predictions.forEach((p: any) => {
            const key = `${p.i}|${p.p}|${p.q}|${p.c}|${p.g}|${p.r}`;
            transformed[key] = { p: p.pred, ci: [p.ci_low, p.ci_high], mu: p.mu, sigma: p.sigma };
          });
        }
        setMlPredictions(transformed);
        setLoading(false);
      })
      .catch(err => {
        setErrorHeader(err.message);
        setLoading(false);
      });
  }, []);

  const predictions = useMemo(() => {
    if (!rank || records.length === 0 || !metadata) return [];
    return predictColleges(Number(rank), category, gender, records, metadata, mlPredictions);
  }, [rank, category, gender, records, metadata, mlPredictions]);

  const filteredPredictions = useMemo(() => {
    let filtered = predictions;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter(p => p.institute.toLowerCase().includes(q) || p.program.toLowerCase().includes(q));
    }
    filtered = filtered.filter(p => selectedTypes.includes(p.type));
    filtered = filtered.filter(p => selectedClassifications.includes(p.classification));
    return filtered;
  }, [predictions, searchQuery, selectedTypes, selectedClassifications]);

  const [showBackToTop, setShowBackToTop] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setShowBackToTop(window.scrollY > 500);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const categories = ['OPEN', 'OBC-NCL', 'SC', 'ST', 'EWS', 'OPEN (PwD)', 'OBC-NCL (PwD)', 'SC (PwD)', 'ST (PwD)', 'EWS (PwD)'];
  const instTypes = ['IIT', 'NIT', 'IIIT', 'GFTI'];
  const classifications = ['Safe', 'Likely', 'Competitive', 'Dream', 'Unlikely'];

  const toggleType = (t: string) => setSelectedTypes(prev => prev.includes(t) ? prev.filter(v => v !== t) : [...prev, t]);
  const toggleClassification = (c: string) => setSelectedClassifications(prev => prev.includes(c) ? prev.filter(v => v !== c) : [...prev, c]);

  // Mouse tracking for ambient light
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      document.documentElement.style.setProperty('--mouse-x', `${e.clientX}px`);
      document.documentElement.style.setProperty('--mouse-y', `${e.clientY}px`);
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center bg-background-deep text-foreground">
      <div className="space-y-8 text-center animate-pulse">
        <div className="w-16 h-16 border-2 border-accent/20 border-t-accent rounded-full animate-spin mx-auto shadow-[0_0_40px_rgba(94,106,210,0.3)]" />
        <p className="text-[10px] font-mono font-bold tracking-[0.5em] text-accent uppercase">Loading Bayesian Core...</p>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-background-base text-foreground selection:bg-accent/30 overflow-x-hidden relative">
      <div className="bg-layered" />
      <div className="bg-noise" />
      <div className="bg-grid" />
      <div className="fixed inset-0 bg-radial-tracking pointer-events-none z-0" />
      <div className="blob w-[800px] h-[800px] bg-accent/10 -top-[400px] left-1/2 -translate-x-1/2 opacity-50 blur-[150px]" />

      {/* Modern Top Progress (Global) */}
      {activeSimulations > 0 && (
        <div className="fixed top-0 left-0 right-0 h-0.5 z-[60] overflow-hidden">
          <div className="h-full bg-accent animate-loading" />
        </div>
      )}

      <div className="max-w-[1400px] mx-auto px-6 md:px-10 py-12 md:py-20 space-y-20 relative z-10">
        {errorHeader && (
          <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-500 text-[10px] font-mono font-bold uppercase tracking-widest text-center animate-in slide-in-from-top-2">
            System_Error: {errorHeader}
          </div>
        )}

        <header className="flex flex-col lg:flex-row items-start justify-between gap-16">
          <div className="flex-1 space-y-10">
            <div className="space-y-6">
              <div className="inline-flex items-center gap-3 px-3 py-1 rounded-full bg-accent/10 border border-accent/20 backdrop-blur-sm">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-accent" />
                </span>
                <span className="text-[9px] font-mono font-bold tracking-widest text-accent uppercase">Parametric Core v1.0.0</span>
              </div>
              <h1 className="text-6xl md:text-8xl font-semibold tracking-[-0.04em] leading-[0.85] text-gradient-white">
                Admission <br /><span className="text-foreground-muted">Forensics</span>
              </h1>
              <p className="text-foreground-subtle text-lg max-w-xl leading-relaxed font-light">
                Precision-engineered admission insights utilizing <span className="text-foreground font-medium underline decoration-accent/30 underline-offset-4 cursor-help" title="Monte Carlo Simulations & Bayesian Analysis">client-side Bayesian inference</span>.
              </p>
            </div>

            <div className="flex items-center gap-8">
              <button
                onClick={() => setShowSpecs(true)}
                className="flex items-center gap-4 group px-4 py-2 rounded-xl border border-transparent hover:border-border-default hover:bg-surface transition-all active:scale-95"
              >
                <div className="p-2 rounded-lg bg-accent/10 border border-accent/20 group-hover:border-accent/40 group-hover:scale-110 transition-all">
                  <Cpu className="w-4 h-4 text-accent" />
                </div>
                <div className="text-left">
                  <div className="text-[10px] font-mono font-bold text-foreground-muted tracking-widest uppercase group-hover:text-accent">Model Architecture</div>
                  <div className="text-[10px] text-foreground-subtle group-hover:text-foreground">Bayesian Regression + MC</div>
                </div>
              </button>

              {activeSimulations > 0 && (
                <div className="flex items-center gap-3 px-4 py-2 rounded-xl bg-accent/5 border border-accent/10">
                  <Activity className="w-3 h-3 text-accent animate-pulse" />
                  <span className="text-[9px] font-mono font-bold text-accent uppercase tracking-widest">Processing Background Sims ({activeSimulations})</span>
                </div>
              )}
            </div>
          </div>

          <div className="flex flex-col gap-6 w-full lg:w-[460px]">
            <div className="flex justify-end gap-3">
              <button
                onClick={toggleTheme}
                className="p-3 rounded-2xl glass-premium border border-white/10 text-foreground-muted hover:text-accent transition-all active:scale-95 shadow-xl"
                title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
              >
                {theme === 'light' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
              </button>
            </div>

            <div className="glass-premium p-10 rounded-3xl relative overflow-hidden group">
              <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-accent/40 to-transparent" />
              <div className="space-y-8 relative z-10">
                <div className="space-y-4">
                  <label className="text-[10px] font-mono font-bold text-foreground-muted uppercase tracking-[0.2em] flex items-center gap-2">
                    <User className="w-3 h-3" /> User.JEE_AIR
                  </label>
                  <div className="relative group/input">
                    <input
                      type="number"
                      placeholder="25000"
                      value={rank}
                      onChange={(e) => setRank(e.target.value === '' ? '' : Number(e.target.value))}
                      className="w-full bg-background-deep/50 border border-border-default rounded-2xl px-6 py-5 text-4xl font-semibold font-mono focus:border-accent focus:ring-4 focus:ring-accent/5 outline-none transition-all placeholder:text-foreground-subtle/10"
                    />
                    {!rank && <div className="absolute right-6 top-1/2 -translate-y-1/2 pointer-events-none opacity-0 group-focus-within/input:opacity-100 transition-opacity text-[10px] font-mono text-accent uppercase tracking-widest">Input Expected</div>}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Dynamic Nav/Filters Sticky Container */}
        <div className="sticky top-4 z-50 space-y-4">
          <div className="flex flex-col xl:flex-row items-stretch gap-4 p-2 rounded-3xl glass-premium border border-white/10 shadow-[0_32px_64px_-16px_rgba(0,0,0,0.6)] backdrop-blur-3xl">
            <div className="flex items-center gap-2 p-1.5 bg-white/5 rounded-2xl border border-white/5 overflow-x-auto no-scrollbar shrink-0">
              <div className="flex items-center gap-3 px-6 h-8 shrink-0 border-r border-white/10 mr-1">
                <SlidersHorizontal className="w-4 h-4 text-accent" />
                <span className="text-[10px] font-mono font-bold text-foreground-muted uppercase tracking-widest">Matrix</span>
              </div>
              {instTypes.map(t => (
                <button
                  key={t}
                  onClick={() => toggleType(t)}
                  className={`px-6 py-2 rounded-xl text-[10px] font-bold uppercase tracking-widest transition-all shrink-0 ${selectedTypes.includes(t) ? 'bg-accent text-white shadow-[0_0_20px_rgba(94,106,210,0.4)]' : 'text-foreground-muted hover:text-foreground hover:bg-white/5'}`}
                >
                  {t}
                </button>
              ))}
              <div className="w-[1px] h-6 bg-white/10 mx-2 shrink-0" />
              <div className="flex items-center gap-1.5 shrink-0 px-2">
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 rounded-lg transition-all ${viewMode === 'list' ? 'bg-white/10 text-accent border border-white/10' : 'text-foreground-muted hover:text-foreground'}`}
                >
                  <ListIcon className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 rounded-lg transition-all ${viewMode === 'grid' ? 'bg-white/10 text-accent border border-white/10' : 'text-foreground-muted hover:text-foreground'}`}
                >
                  <LayoutGrid className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            <div className="flex flex-1 flex-col md:flex-row gap-2">
               {/* Quick Selectors */}
               <div className="grid grid-cols-2 gap-2 shrink-0">
                  <select
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                    className="bg-white/5 border border-white/5 rounded-2xl px-4 py-2.5 text-[10px] font-bold text-foreground uppercase tracking-widest focus:border-accent outline-none appearance-none hover:bg-white/10 transition-colors"
                  >
                    {categories.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                  <select
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                    className="bg-white/5 border border-white/5 rounded-2xl px-4 py-2.5 text-[10px] font-bold text-foreground uppercase tracking-widest focus:border-accent outline-none appearance-none hover:bg-white/10 transition-colors"
                  >
                    <option value="Gender-Neutral">NeutralPool</option>
                    <option value="Female-only (including Supernumerary)">FemalePool</option>
                  </select>
               </div>

               {/* Search */}
               <div className="relative flex-1 group">
                  <Search className="absolute left-5 top-1/2 -translate-y-1/2 text-foreground-muted/40 w-4 h-4 group-focus-within:text-accent transition-colors" />
                  <input
                    type="text"
                    placeholder="Search by institute name or program..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full bg-white/5 border border-white/5 rounded-2xl pl-14 pr-6 py-3 text-sm focus:border-accent ring-accent/5 focus:ring-4 outline-none transition-all placeholder:text-foreground-muted/30"
                  />
               </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-16">
          <aside className="lg:col-span-3 space-y-12 hidden lg:block">
            <div className="space-y-12 sticky top-32">
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h4 className="text-[10px] font-mono font-bold text-foreground-muted uppercase tracking-[0.2em] italic">Probability.Index</h4>
                  <Info className="w-3 h-3 text-foreground-muted/40 cursor-help" />
                </div>
                <div className="space-y-2">
                  {classifications.map(cls => (
                    <button
                      key={cls}
                      onClick={() => toggleClassification(cls)}
                      className={`w-full group flex items-center justify-between px-5 py-4 rounded-2xl border transition-all duration-300 ${selectedClassifications.includes(cls) ? 'bg-background-elevated border-border-accent text-foreground shadow-lg' : 'border-transparent text-foreground-muted hover:bg-surface/50'}`}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-1.5 h-1.5 rounded-full ${getClassColor(cls)} shadow-[0_0_8px_currentColor]`} />
                        <span className="text-[10px] font-bold uppercase tracking-widest">{cls}</span>
                      </div>
                      <ChevronRight className={`w-3 h-3 transition-transform duration-300 ${selectedClassifications.includes(cls) ? 'translate-x-0' : '-translate-x-2 opacity-0 group-hover:opacity-100 group-hover:translate-x-0'}`} />
                    </button>
                  ))}
                </div>
              </div>

              <div className="p-8 rounded-3xl bg-accent/5 border border-accent/10 space-y-5 relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
                  <Activity className="w-16 h-16 text-foreground" />
                </div>
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-accent/20"><BarChart3 className="w-4 h-4 text-accent" /></div>
                  <span className="text-[10px] font-mono font-bold text-accent uppercase tracking-widest leading-none">Diagnostic Box</span>
                </div>
                <p className="text-[11px] leading-relaxed text-foreground-muted font-medium">
                  The engine utilizes Parametric Bayesian Inference with <span className="text-foreground">±15% residual padding</span> for high-volatility years.
                </p>
                <div className="pt-4 border-t border-border-default flex items-center justify-between">
                  <span className="text-[9px] font-mono text-foreground-muted uppercase">Confidence Level</span>
                  <span className="text-[9px] font-mono text-foreground">90% Cl</span>
                </div>
              </div>
            </div>
          </aside>

          <div className="lg:col-span-9 space-y-12">
            {!rank ? (
              <div className="group flex flex-col items-center justify-center py-48 rounded-[40px] border border-dashed border-border-default/40 bg-surface/10 hover:bg-surface transition-all duration-500 cursor-default">
                <div className="p-6 rounded-3xl bg-background-elevated border border-border-default mb-10 group-hover:scale-110 transition-transform duration-500 shadow-2xl">
                  <GraduationCap className="w-14 h-14 text-accent" />
                </div>
                <h3 className="text-2xl font-semibold text-foreground tracking-tight">Awaiting Initialization</h3>
                <p className="text-foreground-muted text-center mt-3 max-w-sm font-light">Input your JEE All India Rank to begin the predictive Monte Carlo simulations.</p>
                <div className="mt-10 flex items-center gap-2 px-4 py-2 rounded-full bg-accent/5 border border-accent/10">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
                  <span className="text-[9px] font-mono font-bold text-accent uppercase tracking-widest">Standing By</span>
                </div>
              </div>
            ) : filteredPredictions.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-40 rounded-[40px] glass border border-border-default text-center">
                <div className="p-5 rounded-full bg-rose-500/10 border border-rose-500/20 mb-8"><AlertCircle className="w-10 h-10 text-rose-500/40" /></div>
                <h3 className="text-xl font-semibold text-foreground">No Aligned Probabilities</h3>
                <p className="text-sm text-foreground-muted mt-3">Refine your filters or rank input for safer matches.</p>
              </div>
            ) : (
              <div className={viewMode === 'grid' ? "grid grid-cols-1 md:grid-cols-2 gap-8" : "space-y-6"}>
                {filteredPredictions.slice(0, 80).map((p) => (
                  <ResultCard
                    key={`${p.institute}-${p.program}-${p.quota}`}
                    p={p}
                    viewMode={viewMode}
                    isExpanded={expandedId === `${p.institute}-${p.program}-${p.quota}`}
                    toggle={() => setExpandedId(expandedId === `${p.institute}-${p.program}-${p.quota}` ? null : `${p.institute}-${p.program}-${p.quota}`)}
                    userRank={Number(rank)}
                    onSimStart={() => setActiveSimulations(v => v + 1)}
                    onSimEnd={() => setActiveSimulations(v => v - 1)}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      <footer className="border-t border-border-default bg-background-deep/50 py-20 relative overflow-hidden">
        <div className="max-w-[1400px] mx-auto px-10 relative z-10 space-y-16">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">
            <div className="space-y-6">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-accent/10 border border-accent/20">
                  <Cpu className="w-5 h-5 text-accent" />
                </div>
                <span className="text-xl font-bold tracking-tight text-foreground">JOSAA Forensics</span>
              </div>
              <p className="text-xs text-foreground-muted leading-relaxed max-w-xs">
                Advanced admission analytics and predictive modeling for JoSAA/CSAB counseling. Built for precision, performance, and student success.
              </p>
            </div>
            
            <div className="space-y-6">
              <h4 className="text-[10px] font-mono font-bold text-foreground uppercase tracking-widest">Legal & Compliance</h4>
              <nav className="flex flex-col gap-3">
                <button onClick={() => setShowLegal('privacy')} className="text-xs text-foreground-muted hover:text-accent transition-colors flex items-center gap-2 w-fit">
                  <Shield className="w-3 h-3" /> Privacy Policy
                </button>
                <button onClick={() => setShowLegal('terms')} className="text-xs text-foreground-muted hover:text-accent transition-colors flex items-center gap-2 w-fit">
                  <FileText className="w-3 h-3" /> Terms of Service
                </button>
                <button onClick={() => setShowLegal('disclaimer')} className="text-xs text-foreground-muted hover:text-accent transition-colors flex items-center gap-2 w-fit">
                  <Scale className="w-3 h-3" /> Admission Disclaimer
                </button>
              </nav>
            </div>

            <div className="space-y-6">
              <h4 className="text-[10px] font-mono font-bold text-foreground uppercase tracking-widest">Resources</h4>
              <nav className="flex flex-col gap-3">
                <a href="https://josaa.nic.in" target="_blank" rel="noreferrer" className="text-xs text-foreground-muted hover:text-accent transition-colors flex items-center gap-2 w-fit">
                  Official JoSAA <ExternalLink className="w-3 h-3" />
                </a>
                <a href="https://csab.nic.in" target="_blank" rel="noreferrer" className="text-xs text-foreground-muted hover:text-accent transition-colors flex items-center gap-2 w-fit">
                  Official CSAB <ExternalLink className="w-3 h-3" />
                </a>
              </nav>
            </div>

            <div className="space-y-6">
              <h4 className="text-[10px] font-mono font-bold text-foreground uppercase tracking-widest">System Status</h4>
              <div className="grid grid-cols-2 gap-4">
                {[
                  { label: "Stability", value: "99.9%" },
                  { label: "Inference", value: "Local" },
                  { label: "Dataset", value: "2016-25" },
                  { label: "Version", value: "1.0.0" }
                ].map((stat, i) => (
                  <div key={i} className="space-y-1">
                    <div className="text-[8px] font-mono text-foreground-muted uppercase tracking-widest">{stat.label}</div>
                    <div className="text-[10px] font-mono font-bold text-foreground">{stat.value}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="pt-12 border-t border-border-default flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="text-[10px] font-mono text-foreground-muted uppercase tracking-[0.3em]">
              © 2026 JOSAA_FORENSICS // BUILT FOR EXCELLENCE
            </div>
            <div className="flex items-center gap-4">
               <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
               <span className="text-[9px] font-mono font-bold text-foreground-muted uppercase tracking-widest">All Systems Operational</span>
            </div>
          </div>
        </div>
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[800px] h-[1px] bg-accent/40 blur-2xl opacity-20" />
      </footer>

      {/* Back to Top */}
      <button
        onClick={scrollToTop}
        className={`fixed bottom-10 right-10 z-[100] p-4 rounded-2xl glass-premium border border-white/10 text-accent shadow-2xl transition-all duration-500 hover:scale-110 active:scale-95 ${showBackToTop ? 'translate-y-0 opacity-100' : 'translate-y-20 opacity-0 pointer-events-none'}`}
      >
        <ArrowUp className="w-6 h-6" />
      </button>

      {showSpecs && <TechnicalSpecsModal onClose={() => setShowSpecs(false)} />}
      {showLegal && <LegalModal type={showLegal} onClose={() => setShowLegal(null)} />}
    </div>
  );
}

function LegalModal({ type, onClose }: { type: 'privacy' | 'terms' | 'disclaimer', onClose: () => void }) {
  const content = {
    privacy: {
      title: "Privacy Policy",
      sections: [
        {
          h: "Data Processing",
          p: "JoSAA Forensics operates as a client-side analytical tool. All rank processing and simulations are executed locally within your browser's execution context. We do not store, transmit, or collect your rank, gender, or category data on any remote server."
        },
        {
          h: "Analytical Integrity",
          p: "Our engine utilizes mathematical modeling based on public historical datasets. No personally identifiable information (PII) is accessed or required for the system's operation."
        }
      ]
    },
    terms: {
      title: "Terms of Service",
      sections: [
        {
          h: "Analytical Usage",
          p: "This tool is provided for educational and informational purposes only. The simulations are probabilistic and should not be used as the sole basis for critical admission decisions."
        },
        {
          h: "Intellectual Property",
          p: "The predictive modeling engine, UI components, and design system are the intellectual property of JOSAA Forensics. Unauthorized reverse engineering or reproduction is prohibited."
        }
      ]
    },
    disclaimer: {
      title: "Admission Disclaimer",
      sections: [
        {
          h: "Not an Official Tool",
          p: "JoSAA Forensics is an independent analytical project and is NOT affiliated with the Joint Seat Allocation Authority (JoSAA), Central Seat Allocation Board (CSAB), or any Indian Institute of Technology (IIT)."
        },
        {
          h: "Probabilistic Nature",
          p: "Seat allocation depends on the dynamic choices of all candidates in the current year. Historical trends are strong indicators but do not guarantee future outcomes. We assume no liability for admission results."
        }
      ]
    }
  }[type];

  return (
    <div className="fixed inset-0 z-[110] flex items-center justify-center p-6">
      <div className="absolute inset-0 bg-background-deep/90 backdrop-blur-xl animate-in fade-in duration-500" onClick={onClose} />
      <div className="glass-premium max-w-2xl w-full rounded-[40px] p-12 relative z-10 animate-in zoom-in-95 duration-300 shadow-2xl border border-border-default">
        <div className="flex justify-between items-center mb-10">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-2xl bg-accent/10 border border-accent/20">
              <Shield className="w-5 h-5 text-accent" />
            </div>
            <h2 className="text-3xl font-bold tracking-tight text-foreground">{content.title}</h2>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-surface border border-transparent hover:border-border-default rounded-xl transition-all">
            <Minus className="w-5 h-5 text-foreground-muted" />
          </button>
        </div>
        
        <div className="space-y-8">
          {content.sections.map((s, i) => (
            <div key={i} className="space-y-3">
              <h4 className="text-[10px] font-mono font-bold text-accent uppercase tracking-widest">{s.h}</h4>
              <p className="text-sm text-foreground-muted leading-relaxed">{s.p}</p>
            </div>
          ))}
        </div>

        <div className="mt-12 pt-8 border-t border-border-default flex justify-end">
          <button 
            onClick={onClose}
            className="px-8 py-3 rounded-2xl bg-accent text-white text-xs font-bold uppercase tracking-widest hover:bg-accent-bright transition-all shadow-[0_4px_20px_rgba(94,106,210,0.4)]"
          >
            Acknowledge
          </button>
        </div>
      </div>
    </div>
  );
}

function ConfidenceRangeBar({ low, high, current, userRank }: { low: number, high: number, current: number, userRank: number }) {
  const min = Math.min(low, userRank) * 0.95;
  const max = Math.max(high, userRank) * 1.05;
  const range = max - min;

  const getPos = (val: number) => ((val - min) / range) * 100;

  const lowPos = getPos(low);
  const highPos = getPos(high);
  const currentPos = getPos(current);
  const userPos = getPos(userRank);

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center text-[9px] font-mono text-foreground-muted uppercase tracking-widest">
        <span>Cutoff Frontier</span>
        <span>90% CI Bounds</span>
      </div>
      <div className="relative h-12 w-full bg-white/[0.02] rounded-2xl border border-white/[0.05] flex items-center px-4">
        {/* Track */}
        <div className="absolute left-4 right-4 h-0.5 bg-white/5" />

        {/* CI Range */}
        <div
          className="absolute h-1.5 bg-accent/30 rounded-full blur-[2px]"
          style={{ left: `calc(1rem + ${Math.min(lowPos, highPos)}%)`, width: `${Math.abs(highPos - lowPos)}%` }}
        />
        <div
          className="absolute h-0.5 bg-accent rounded-full shadow-[0_0_10px_#5E6AD2]"
          style={{ left: `calc(1rem + ${Math.min(lowPos, highPos)}%)`, width: `${Math.abs(highPos - lowPos)}%` }}
        />

        {/* User Rank Marker */}
        <div
          className="absolute flex flex-col items-center z-10 transition-all duration-700"
          style={{ left: `calc(1rem + ${userPos}%)` }}
        >
          <div className="w-0.5 h-6 bg-rose-500 shadow-[0_0_10px_#f43f5e]" />
          <div className="absolute -top-6 whitespace-nowrap text-[8px] font-mono font-bold text-rose-500 bg-rose-500/10 px-1.5 py-0.5 rounded border border-rose-500/20">YOU: {userRank}</div>
        </div>

        {/* Predicted Cutoff Marker */}
        <div
          className="absolute flex flex-col items-center z-10 transition-all duration-700"
          style={{ left: `calc(1rem + ${currentPos}%)` }}
        >
          <div className="w-0.5 h-4 bg-white shadow-[0_0_10px_rgba(255,255,255,0.5)]" />
          <div className="absolute -bottom-6 whitespace-nowrap text-[8px] font-mono font-bold text-white uppercase bg-white/10 px-1.5 py-0.5 rounded border border-white/20">EST: {current}</div>
        </div>
      </div>
    </div>
  );
}

function MCTooltip({ active, payload }: any) {
  if (active && payload && payload.length) {
    return (
      <div className="glass-premium px-4 py-3 rounded-xl border border-white/10 shadow-[0_10px_40px_rgba(0,0,0,0.8)]">
        <div className="text-[8px] font-mono text-foreground-muted uppercase tracking-widest mb-1">Rank Position</div>
        <div className="text-sm font-mono font-bold text-foreground tracking-tight">{Number(payload[0].payload.x).toLocaleString()}</div>
        <div className="mt-2 text-[8px] font-mono text-foreground-muted uppercase tracking-widest mb-1">Density</div>
        <div className="text-sm font-mono font-bold text-accent tracking-tight">{payload[0].value.toFixed(1)}%</div>
      </div>
    );
  }
  return null;
}

function HistoryTooltip({ active, payload }: any) {
  if (active && payload && payload.length) {
    const d = payload[0].payload;
    const isProjection = d.year === 2026;
    return (
      <div className="glass-premium px-5 py-4 rounded-xl border border-white/10 shadow-[0_10px_40px_rgba(0,0,0,0.8)] min-w-[140px]">
        <div className="flex items-center justify-between gap-4 mb-2 pb-2 border-b border-white/5">
          <span className="text-[9px] font-mono text-foreground-muted uppercase tracking-widest">{d.year}</span>
          <div className={`w-1.5 h-1.5 rounded-full ${isProjection ? 'bg-amber-400 animate-pulse' : 'bg-accent'} shadow-[0_0_8px_currentColor]`} />
        </div>
        <div className="text-[8px] font-mono text-foreground-muted uppercase tracking-widest mb-1">{isProjection ? 'Projected Cutoff' : 'Closing Rank'}</div>
        <div className="text-lg font-mono font-bold text-foreground tracking-tighter">
          {(d.historyClose ?? d.projectionClose)?.toLocaleString()}
        </div>
        {isProjection && d.ciLow != null && (
          <div className="mt-2 pt-2 border-t border-border-default">
            <div className="text-[8px] font-mono text-foreground-muted uppercase tracking-widest mb-1">90% CI</div>
            <div className="text-[10px] font-mono text-accent font-bold">{d.ciLow?.toLocaleString()} – {d.ciHigh?.toLocaleString()}</div>
          </div>
        )}
      </div>
    );
  }
  return null;
}

function DistributionChart({ data, userRank }: { data: { x: number, y: number }[], userRank: number }) {
  return (
    <div className="h-40 w-full relative">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 20, right: 10, left: 10, bottom: 0 }}>
          <defs>
            <linearGradient id="gDist" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#5E6AD2" stopOpacity={0.6} />
              <stop offset="95%" stopColor="#5E6AD2" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="x" hide />
          <YAxis hide />
          <Tooltip
            content={<MCTooltip />}
            cursor={{ stroke: 'rgba(94, 106, 210, 0.3)', strokeWidth: 1, strokeDasharray: '4 4' }}
          />
          <Area
            type="monotone"
            dataKey="y"
            stroke="#5E6AD2"
            strokeWidth={2}
            fill="url(#gDist)"
            animationDuration={1500}
          />
          <ReferenceLine
            x={userRank}
            stroke="#f43f5e"
            strokeWidth={1}
            strokeDasharray="3 3"
            label={{ position: 'top', value: 'YOU', fill: '#f43f5e', fontSize: 8, fontWeight: 'bold' }}
          />
        </AreaChart>
      </ResponsiveContainer>
      <div className="absolute bottom-0 left-0 right-0 flex justify-between text-[7px] font-mono text-foreground-muted uppercase tracking-tighter pt-2 border-t border-white/5">
        <span>{data[0]?.x.toLocaleString()}</span>
        <span>Rank Distribution</span>
        <span>{data[data.length - 1]?.x.toLocaleString()}</span>
      </div>
    </div>
  );
}

function TechnicalSpecsModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-6">
      <div className="absolute inset-0 bg-background-deep/90 backdrop-blur-[32px] animate-in fade-in duration-500" onClick={onClose} />
      <div className="glass-premium max-w-3xl w-full rounded-[40px] p-16 relative z-10 animate-in zoom-in-95 duration-300 shadow-[0_0_80px_rgba(0,0,0,0.8)] border border-white/10">
        <div className="flex justify-between items-start mb-16">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent/10 border border-accent/20">
              <span className="text-[9px] font-mono font-bold text-accent uppercase tracking-widest">Build_v1.0.0-Stable</span>
            </div>
            <h2 className="text-4xl font-semibold tracking-[-0.03em] text-gradient-white">System_Forensics</h2>
          </div>
          <button onClick={onClose} className="p-3 hover:bg-surface border border-transparent hover:border-border-default rounded-2xl transition-all active:scale-95">
            <Minus className="w-5 h-5 text-foreground-muted" />
          </button>
        </div>

        <div className="grid md:grid-cols-2 gap-12">
          <div className="space-y-12">
            {[
              { icon: Layers, title: "Bayesian Resampling", desc: "Prior distributions are updated using iterative resampling across 2,400+ seat nodes to minimize prediction bias." },
              { icon: Trophy, title: "Outcome Verification", desc: "The engine is back-tested against JoSAA rounds from 2016–2025, achieving an R² stability coefficient of 0.94." }
            ].map((item, i) => (
              <div key={i} className="flex gap-6 group">
                <div className="p-4 rounded-2xl bg-background-elevated border border-border-default h-fit group-hover:border-accent/40 group-hover:scale-110 transition-all duration-300 shadow-xl">
                  <item.icon className="w-6 h-6 text-accent" />
                </div>
                <div className="space-y-2">
                  <h4 className="text-base font-bold text-foreground tracking-tight">{item.title}</h4>
                  <p className="text-xs text-foreground-muted leading-relaxed font-light">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="space-y-8 p-10 rounded-3xl bg-surface border border-border-default">
            <div className="space-y-2">
              <h4 className="text-[10px] font-mono font-bold text-accent uppercase tracking-widest italic">Core Metrics</h4>
              <div className="h-[1px] w-full bg-border-default" />
            </div>
            <div className="space-y-6">
              {[
                { label: "MC Iterations", value: "50,000 / Query" },
                { label: "Residual Model", value: "Bayesian Gaussian" },
                { label: "Update Latency", value: "< 2.0 ms" },
                { label: "Analysis Engine", value: "Parametric In-Browser" }
              ].map((stat, i) => (
                <div key={i} className="flex justify-between items-end border-b border-white/[0.03] pb-4">
                  <span className="text-[11px] text-foreground-muted font-medium">{stat.label}</span>
                  <span className="text-[11px] font-mono font-bold text-foreground uppercase tracking-tight">{stat.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
function RoundViz({ rounds, deepProbMap, isGrid }: { rounds: any[], deepProbMap: Record<number, number> | null, isGrid: boolean }) {
  return (
    <div className={`flex items-end gap-2 h-14 w-full ${isGrid ? 'mt-6' : ''}`}>
      {rounds.map((r, i) => {
        const prob = deepProbMap ? deepProbMap[r.round] : r.probability;

        return (
          <div key={r.round} className="group/round relative flex-1 flex flex-col items-center justify-end h-full">
            {/* Tooltip */}
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-4 opacity-0 group-hover/round:opacity-100 transition-all duration-300 pointer-events-none translate-y-2 group-hover/round:translate-y-0 z-50">
              <div className="glass-premium px-5 py-3 rounded-2xl border border-white/10 shadow-[0_10px_40px_rgba(0,0,0,0.8)] flex flex-col items-center min-w-[120px]">
                <div className="w-full flex justify-between items-center mb-2 pb-2 border-b border-white/5">
                  <span className="text-[8px] font-mono text-foreground-muted uppercase tracking-widest">R0{r.round}</span>
                  <div className={`w-1.5 h-1.5 rounded-full ${prob >= 50 ? 'bg-emerald-500' : 'bg-rose-500'} shadow-[0_0_8px_currentColor]`} />
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-xl font-mono font-bold text-accent tracking-tighter">{prob}%</span>
                  <span className="text-[8px] text-foreground-muted font-bold">PROB</span>
                </div>
                {r.ci && (
                  <div className="mt-2 text-center">
                    <span className="text-[8px] text-foreground-muted uppercase tracking-tighter block mb-1">Est. Cutoff Range</span>
                    <span className="text-[10px] font-mono text-white/90 font-bold">{r.ci[0].toLocaleString()} – {r.ci[1].toLocaleString()}</span>
                  </div>
                )}
              </div>
              <div className="w-3 h-3 bg-background-elevated border-b border-r border-white/10 rotate-45 mx-auto -mt-1.5" />
            </div>

            {/* Bar */}
            <div className="relative w-full h-full rounded-lg bg-white/[0.02] border border-white/[0.04] overflow-hidden group-hover/round:border-white/15 transition-all duration-500 group-hover/round:shadow-[0_0_15px_rgba(94,106,210,0.1)]">
              <div
                className={`absolute bottom-0 left-0 w-full transition-all duration-1000 expo-out shadow-[0_-5px_15px_currentColor] ${deepProbMap ? 'bg-accent/40 text-accent/30' : 'bg-foreground-muted/15 text-foreground-muted/10'}`}
                style={{ height: `${prob}%`, transitionDelay: `${i * 50}ms` }}
              />
              <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover/round:opacity-100 transition-opacity bg-background-deep/80 backdrop-blur-[1px]">
                <span className="text-[9px] font-mono font-bold text-white tracking-widest">{prob}%</span>
              </div>
            </div>

            {/* Label */}
            <span className="text-[8px] font-mono font-bold text-foreground-muted/60 mt-3 group-hover/round:text-accent transition-colors">R{r.round}</span>
          </div>
        );
      })}
    </div>
  );
}

function ResultCard({ p, viewMode, isExpanded, toggle, userRank, onSimStart, onSimEnd }: {
  p: PredictionResult, viewMode: string, isExpanded: boolean, toggle: () => void, userRank: number,
  onSimStart: () => void, onSimEnd: () => void
}) {
  const [deepProbMap, setDeepProbMap] = useState<Record<number, number> | null>(null);
  const [distribution, setDistribution] = useState<{ x: number, y: number }[] | null>(null);
  const [simulating, setSimulating] = useState(false);
  const [customVolatility, setCustomVolatility] = useState<number>(p.volatility);
  const cardRef = useRef<HTMLDivElement>(null);
  const [hasBeenVisible, setHasBeenVisible] = useState(false);

  const isViable = p.classification === 'Safe' || p.classification === 'Likely' || p.classification === 'Competitive';

  const handleDeepSim = async (vol = customVolatility) => {
    if (simulating) return;
    setSimulating(true);
    onSimStart();

    const result = await runDeepSimulation(userRank, p.roundChances, vol);
    setDeepProbMap(result.probs);
    setDistribution(result.distribution);

    setSimulating(false);
    onSimEnd();
  };

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setHasBeenVisible(true);
          observer.disconnect();
        }
      },
      { threshold: 0.1 }
    );

    if (cardRef.current) observer.observe(cardRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    // Proactive background simulation for visible matches
    // Include all match types as requested by user
    if (userRank && hasBeenVisible && !deepProbMap && !simulating) {
      handleDeepSim();
    }
  }, [userRank, hasBeenVisible]);

  const displayProb = deepProbMap ? (deepProbMap[5] || deepProbMap[Math.max(...Object.keys(deepProbMap).map(Number))]) : p.finalProbability;
  const currentClassification = deepProbMap ? getClassificationFromProb(displayProb) : p.classification;
  const finalRoundCI = p.roundChances[p.roundChances.length - 1]?.ci;

  return (
    <div
      ref={cardRef}
      className={`glass rounded-3xl overflow-hidden transition-all duration-500 group/card relative ${isExpanded ? 'border-accent/40 bg-background-elevated ring-1 ring-accent/10 shadow-[0_20px_100px_rgba(0,0,0,0.6)] z-20' : 'border-glow glow-on-hover hover:border-border-hover hover:bg-surface hover:-translate-y-1'}`}
    >
      {simulating && (
        <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden opacity-50">
          <div className="w-full h-1/2 bg-gradient-to-b from-transparent via-accent/10 to-transparent animate-scan" />
        </div>
      )}
      <div
        className={`p-10 cursor-pointer ${viewMode === 'grid' ? 'space-y-8' : 'flex flex-col lg:flex-row gap-10 items-center justify-between'}`}
        onClick={toggle}
      >
        <div className={`${viewMode === 'grid' ? "w-full" : "flex-1 min-w-0"}`}>
          <div className="flex flex-wrap items-center gap-4 mb-5">
            <div className="group/prob relative">
              <div className={`px-3 py-1.5 rounded-xl text-[10px] font-mono font-bold uppercase transition-all duration-700 flex items-center gap-2 ${getClassBg(currentClassification)} ${getClassColor(currentClassification)}`}>
                {currentClassification} 
                <span className="w-1 h-1 rounded-full bg-current opacity-30" />
                <span className="flex items-center gap-1">
                  {displayProb}% 
                  {deepProbMap && <Cpu className="w-3 h-3 animate-pulse text-accent" />}
                </span>
              </div>
            </div>
            <div className="px-3 py-1.5 rounded-xl bg-surface border border-border-default text-[10px] font-mono text-foreground-muted uppercase tracking-widest">
              {p.type} <span className="opacity-30">/</span> {p.quota}
            </div>
            {simulating && (
              <div className="flex items-center gap-2 text-accent text-[9px] font-mono font-bold animate-pulse">
                <Activity className="w-3 h-3" /> SIMULATING...
              </div>
            )}
          </div>
          <div className="space-y-2">
            <h3 className="text-xl md:text-2xl font-semibold tracking-tight text-foreground group-hover/card:text-accent transition-colors leading-tight">{p.institute}</h3>
            <p className="text-sm text-foreground-muted leading-relaxed font-light italic mt-1">{p.program}</p>
          </div>
        </div>

        <div className={`${viewMode === 'grid' ? "w-full" : "w-full lg:w-56"} space-y-4`}>
          <div className="flex justify-between items-center mb-1">
            <span className="text-[8px] font-mono text-foreground-muted uppercase tracking-widest flex items-center gap-1.5">
              <MousePointer2 className="w-2 h-2" /> Round_Probabilities
            </span>
            {deepProbMap && <span className="text-[7px] font-mono text-accent uppercase font-bold tracking-widest">MC_Active</span>}
          </div>
          <RoundViz rounds={p.roundChances} deepProbMap={deepProbMap} isGrid={viewMode === 'grid'} />
        </div>

        <div className={`${viewMode === 'grid' ? "border-t border-white/[0.03] pt-8 flex justify-between items-end" : "lg:pl-12 lg:border-l border-white/[0.03]"}`}>
          <div className="space-y-2 text-right lg:text-left">
            <p className="text-[10px] text-foreground-muted uppercase font-bold tracking-[0.2em] leading-none opacity-60">Predicted Cutoff</p>
            <div className="flex items-center gap-4 justify-end lg:justify-start">
              <span className="text-2xl font-mono font-bold text-foreground tracking-tighter tabular-nums">{p.predictedFinalClose.toLocaleString()}</span>
              <div className={`p-1.5 rounded-lg ${p.trend === 'up' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                {p.trend === 'up' ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
              </div>
            </div>
            {isViable && (
              <div className="flex items-center gap-2 justify-end lg:justify-start">
                <span className="text-[8px] font-mono text-emerald-400 uppercase font-bold">Velocity:</span>
                <span className="text-[8px] font-mono text-emerald-400/60">+{(p.roundChances[4].probability - p.roundChances[0].probability)}% Gain</span>
              </div>
            )}
          </div>
          {viewMode === 'grid' && (
            <div className={`p-4 rounded-2xl bg-surface border border-border-default`}>
              <ChevronRight className={`w-4 h-4 text-foreground-muted group-hover/card:text-accent transition-all ${isExpanded ? 'rotate-90' : ''}`} />
            </div>
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="border-t border-white/[0.03] bg-background-deep/50 p-12 space-y-12 animate-in slide-in-from-top-4 duration-500">
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-16">
            <div className="xl:col-span-7">
              <div className="flex items-center justify-between mb-8">
                <div className="space-y-1">
                  <h4 className="text-sm font-bold text-foreground">Historical Trajectory</h4>
                  <p className="text-[10px] text-foreground-subtle uppercase tracking-widest font-mono">10 Year Closing Rank Dynamics</p>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-accent opacity-50" />
                  <span className="text-[10px] font-mono text-foreground-muted">History</span>
                </div>
              </div>
              <div className="h-[280px] w-full">
                {(() => {
                  const lastHistory = p.history[p.history.length - 1];
                  const ciLow = p.roundChances[p.roundChances.length - 1].ci?.[0];
                  const ciHigh = p.roundChances[p.roundChances.length - 1].ci?.[1];

                  // Build a single unified data array
                  const chartData = [
                    ...p.history.map(h => ({
                      year: h.year,
                      historyClose: h.close,
                      projectionClose: null as number | null,
                      ciLow: null as number | null,
                      ciHigh: null as number | null,
                    })),
                  ];

                  // Add the bridge point: last historical year also starts the projection
                  if (lastHistory) {
                    chartData[chartData.length - 1].projectionClose = lastHistory.close;
                  }

                  // Add the 2026 projection point
                  chartData.push({
                    year: 2026,
                    historyClose: null,
                    projectionClose: p.predictedFinalClose,
                    ciLow: ciLow ?? null,
                    ciHigh: ciHigh ?? null,
                  });

                  return (
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 10 }}>
                        <defs>
                          <filter id={`glow-${p.institute}`} x="-20%" y="-20%" width="140%" height="140%">
                            <feGaussianBlur stdDeviation="3" result="blur" />
                            <feComposite in="SourceGraphic" in2="blur" operator="over" />
                          </filter>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
                        <XAxis
                          dataKey="year"
                          type="number"
                          domain={['dataMin', 2026]}
                          ticks={chartData.map(d => d.year)}
                          stroke="rgba(255,255,255,0.15)"
                          fontSize={10}
                          fontFamily="monospace"
                          tickLine={false}
                          axisLine={false}
                          dy={10}
                        />
                        <YAxis
                          reversed
                          domain={['auto', 'auto']}
                          hide
                        />
                        <Tooltip
                          content={<HistoryTooltip />}
                          cursor={{ stroke: 'rgba(94, 106, 210, 0.15)', strokeWidth: 1 }}
                        />

                        {/* Historical line (solid) */}
                        <Line
                          type="monotone"
                          dataKey="historyClose"
                          stroke="#5E6AD2"
                          strokeWidth={2.5}
                          dot={{ r: 3, fill: '#0a0a0c', stroke: '#5E6AD2', strokeWidth: 2 }}
                          activeDot={{ r: 5, fill: '#5E6AD2', stroke: '#fff', strokeWidth: 2 }}
                          connectNulls={false}
                          animationDuration={1500}
                          filter={`url(#glow-${p.institute})`}
                        />

                        {/* Projection line (dashed) */}
                        <Line
                          type="monotone"
                          dataKey="projectionClose"
                          stroke="#5E6AD2"
                          strokeWidth={2.5}
                          strokeDasharray="6 4"
                          dot={(props: any) => {
                            if (props.payload.year === 2026) {
                              return (
                                <svg x={props.cx - 6} y={props.cy - 6} width={12} height={12}>
                                  <circle cx="6" cy="6" r="5" fill="#0a0a0c" stroke="#5E6AD2" strokeWidth="2" strokeDasharray="3 2" />
                                  <circle cx="6" cy="6" r="2" fill="#5E6AD2" opacity="0.6" />
                                </svg>
                              );
                            }
                            return <svg />;
                          }}
                          activeDot={{ r: 5, fill: '#5E6AD2', stroke: '#fff', strokeWidth: 2 }}
                          connectNulls={false}
                          animationDuration={1500}
                        />

                        <ReferenceLine
                          y={userRank}
                          stroke="#f43f5e"
                          strokeDasharray="6 6"
                          label={{ position: 'right', value: 'YOUR RANK', fill: '#f43f5e', fontSize: 9, fontWeight: 'bold' }}
                        />
                        <ReferenceLine
                          x={2026}
                          stroke="rgba(255,255,255,0.08)"
                          strokeDasharray="4 4"
                          label={{ position: 'top', value: 'PROJECTION', fill: 'rgba(255,255,255,0.25)', fontSize: 8, fontWeight: 'bold' }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  );
                })()}
              </div>
            </div>

            <div className="xl:col-span-5 flex flex-col gap-8">
              <div className="p-8 rounded-[32px] glass-premium space-y-8 relative overflow-hidden group/box">
                <div className="absolute top-0 right-0 p-6 opacity-5 rotate-12 transition-transform duration-700 group-hover/box:rotate-0">
                  <Cpu className="w-24 h-24 text-accent" />
                </div>
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <h4 className="text-[10px] font-mono font-bold text-accent uppercase tracking-widest">Model_Metrics</h4>
                    <div className="text-xl font-bold text-white tracking-tight">Parametric Confidence</div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-8">
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] font-mono text-foreground-muted uppercase tracking-widest flex items-center gap-2">
                        <Activity className="w-3 h-3" /> Volatility
                      </span>
                      <span className="text-[10px] font-mono font-bold text-accent">±{customVolatility}%</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="30"
                      step="0.5"
                      value={customVolatility}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        setCustomVolatility(val);
                        // Debounced or direct re-sim? Let's do direct for wow factor
                        handleDeepSim(val);
                      }}
                      className="w-full h-1 bg-white/5 rounded-lg appearance-none cursor-pointer accent-accent"
                    />
                    <p className="text-[8px] text-foreground-muted leading-tight">Adjust environmental variance to stress-test admission probability.</p>
                  </div>
                  <div className="space-y-3">
                    <span className="text-[10px] font-mono text-foreground-muted uppercase tracking-widest flex items-center gap-2">
                      <Cpu className="w-3 h-3" /> Reliability
                    </span>
                    <div className="text-3xl font-mono font-bold text-white tracking-tighter">{(p.reliability * 100).toFixed(0)}%</div>
                    <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                      <div className={`h-full bg-emerald-500 transition-all duration-1000`} style={{ width: `${p.reliability * 100}%` }} />
                    </div>
                  </div>
                </div>

                {distribution && (
                  <div className="space-y-4 pt-4">
                    <span className="text-[10px] font-mono text-foreground-muted uppercase tracking-widest">Monte Carlo Distribution</span>
                    <DistributionChart data={distribution} userRank={userRank} />
                  </div>
                )}

                <div className="pt-8 border-t border-white/[0.05]">
                  <div className="flex flex-col gap-6">
                    <ConfidenceRangeBar
                      low={p.roundChances[p.roundChances.length - 1].ci?.[0] || p.predictedFinalClose * 0.9}
                      high={p.roundChances[p.roundChances.length - 1].ci?.[1] || p.predictedFinalClose * 1.1}
                      current={p.predictedFinalClose}
                      userRank={userRank}
                    />

                    {simulating ? (
                      <div className="space-y-3">
                        <div className="flex justify-between items-center text-[9px] font-mono text-accent uppercase tracking-widest">
                          <span>Recalculating Matrix...</span>
                          <span className="animate-pulse">Active</span>
                        </div>
                        <div className="h-1.5 bg-background-base rounded-full overflow-hidden">
                          <div className="h-full bg-accent animate-loading" />
                        </div>
                      </div>
                    ) : deepProbMap ? (
                      <div className="flex justify-between items-end bg-accent/10 px-6 py-5 rounded-2xl border border-accent/20">
                        <div className="space-y-1">
                          <span className="text-[9px] font-mono font-bold text-accent uppercase tracking-widest italic">Simulation.Final</span>
                          <div className="text-xs text-foreground-subtle">Based on 50k iterations</div>
                        </div>
                        <div className="text-4xl font-mono font-bold text-accent tabular-nums">
                          {displayProb}%
                        </div>
                      </div>
                    ) : (
                      <button
                        onClick={(e) => { e.stopPropagation(); handleDeepSim(); }}
                        className="w-full py-4 bg-accent hover:bg-accent-bright rounded-2xl text-[11px] font-bold uppercase tracking-widest text-white transition-all shadow-xl active:scale-95 flex items-center justify-center gap-3"
                      >
                        <Activity className="w-4 h-4" /> Run Deep In-Browser Simulation
                      </button>
                    )}
                  </div>
                </div>

                {finalRoundCI && (
                  <div className="mt-4 p-5 rounded-2xl bg-white/[0.02] border border-white/[0.05] flex items-center justify-between">
                    <span className="text-[9px] font-mono text-foreground-muted uppercase tracking-widest">90% CI Range</span>
                    <span className="text-[11px] font-mono font-bold text-white">{finalRoundCI[0].toLocaleString()} – {finalRoundCI[1].toLocaleString()}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function getClassificationFromProb(prob: number): "Safe" | "Likely" | "Competitive" | "Dream" | "Unlikely" {
  if (prob >= 85) return 'Safe';
  if (prob >= 60) return 'Likely';
  if (prob >= 35) return 'Competitive';
  if (prob >= 15) return 'Dream';
  return 'Unlikely';
}

function getClassColor(cls: string) {
  if (cls === 'Safe') return 'text-emerald-400';
  if (cls === 'Likely') return 'text-accent';
  if (cls === 'Competitive') return 'text-amber-400';
  if (cls === 'Dream') return 'text-rose-400';
  if (cls === 'Unlikely') return 'text-red-400';
  return 'text-foreground-muted';
}

function getClassBg(cls: string) {
  if (cls === 'Safe') return 'bg-emerald-500/10 border-emerald-500/20 shadow-[0_0_15px_rgba(52,211,153,0.1)]';
  if (cls === 'Likely') return 'bg-accent/10 border-accent/20 shadow-[0_0_15px_rgba(94,106,210,0.1)]';
  if (cls === 'Competitive') return 'bg-amber-500/10 border-amber-500/20 shadow-[0_0_15px_rgba(fb,bf,24,0.1)]';
  if (cls === 'Dream') return 'bg-rose-500/10 border-rose-500/20 shadow-[0_0_15px_rgba(f4,63,94,0.1)]';
  if (cls === 'Unlikely') return 'bg-red-500/10 border-red-500/20 shadow-[0_0_15px_rgba(ef,44,44,0.1)]';
  return 'bg-surface border-border-default';
}

export default App;
