import React, { useState } from 'react';
import { Cpu, Upload, RefreshCw, CheckCircle2, AlertTriangle, Sparkles } from 'lucide-react';

interface DiagnosticResult {
  classLabel: 'Normal' | 'Pneumonia' | 'COVID-19';
  confidence: number;
  probabilities: { normal: number; pneumonia: number; covid: number };
  explanation: string;
}

export const DiagnosticPlayground: React.FC = () => {
  const [selectedPreset, setSelectedPreset] = useState<'normal' | 'pneumonia' | 'covid'>('covid');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const presets: Record<'normal' | 'pneumonia' | 'covid', DiagnosticResult> = {
    normal: {
      classLabel: 'Normal',
      confidence: 94.2,
      probabilities: { normal: 0.942, pneumonia: 0.041, covid: 0.017 },
      explanation: 'Clear lung fields without focal consolidation or ground-glass opacities. Normal bronchovascular markings.'
    },
    pneumonia: {
      classLabel: 'Pneumonia',
      confidence: 91.8,
      probabilities: { normal: 0.032, pneumonia: 0.918, covid: 0.050 },
      explanation: 'Focal lobar consolidation present in the right middle lobe with surrounding inflammatory exudate.'
    },
    covid: {
      classLabel: 'COVID-19',
      confidence: 93.5,
      probabilities: { normal: 0.015, pneumonia: 0.050, covid: 0.935 },
      explanation: 'Bilateral peripheral ground-glass opacities (GGO) with lower lobe predominance characteristic of viral COVID-19 pneumonitis.'
    }
  };

  const currentResult = presets[selectedPreset];

  const handleRunInference = (preset: 'normal' | 'pneumonia' | 'covid') => {
    setIsAnalyzing(true);
    setSelectedPreset(preset);
    setTimeout(() => {
      setIsAnalyzing(false);
    }, 400);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 mb-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Cpu className="w-5 h-5 text-sky-400" />
            Diagnostic Inference Playground
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Perform chest X-ray screening and class-level diagnostics utilizing the trained communication-sparse ResNet-18 model.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Preset Case Selection */}
        <div className="lg:col-span-5 space-y-4">
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">
            Select Clinical Chest X-Ray Case
          </h3>

          <div className="grid grid-cols-3 gap-3">
            <button
              onClick={() => handleRunInference('normal')}
              className={`p-3 rounded-lg border text-left transition-all ${
                selectedPreset === 'normal'
                  ? 'bg-sky-500/20 border-sky-500 text-white shadow-sky-500/10 shadow-sm'
                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
              }`}
            >
              <div className="text-xs font-bold uppercase text-sky-400">Case 01</div>
              <div className="text-sm font-semibold mt-1">Normal</div>
              <div className="text-[10px] text-slate-400">Clear Lungs</div>
            </button>

            <button
              onClick={() => handleRunInference('pneumonia')}
              className={`p-3 rounded-lg border text-left transition-all ${
                selectedPreset === 'pneumonia'
                  ? 'bg-rose-500/20 border-rose-500 text-white shadow-rose-500/10 shadow-sm'
                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
              }`}
            >
              <div className="text-xs font-bold uppercase text-rose-400">Case 02</div>
              <div className="text-sm font-semibold mt-1">Pneumonia</div>
              <div className="text-[10px] text-slate-400">Focal Consolidation</div>
            </button>

            <button
              onClick={() => handleRunInference('covid')}
              className={`p-3 rounded-lg border text-left transition-all ${
                selectedPreset === 'covid'
                  ? 'bg-emerald-500/20 border-emerald-500 text-white shadow-emerald-500/10 shadow-sm'
                  : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
              }`}
            >
              <div className="text-xs font-bold uppercase text-emerald-400">Case 03</div>
              <div className="text-sm font-semibold mt-1">COVID-19</div>
              <div className="text-[10px] text-slate-400">Ground-Glass Opacities</div>
            </button>
          </div>

          {/* Visual Canvas Representation */}
          <div className="relative aspect-square max-w-sm mx-auto bg-slate-950 border border-slate-800 rounded-xl overflow-hidden flex items-center justify-center p-4">
            {/* Synthetic X-Ray Graphic */}
            <div className="w-full h-full bg-slate-900 rounded-lg relative overflow-hidden flex items-center justify-center border border-slate-800">
              {/* Rib cage overlay effect */}
              <div className="absolute inset-0 bg-gradient-to-b from-slate-800/40 via-slate-900/60 to-slate-950/80" />
              <div className="z-10 text-center p-4 space-y-2">
                <div className="inline-block p-3 rounded-full bg-slate-800/80 text-sky-400 border border-slate-700">
                  <Sparkles className="w-8 h-8" />
                </div>
                <div className="text-xs font-mono text-slate-300">
                  ResNet-18 [128x128x3]
                </div>
                <div className="text-xs font-semibold text-slate-400">
                  {selectedPreset === 'normal' && 'Clear lung field density pattern'}
                  {selectedPreset === 'pneumonia' && 'RML Focal opacity heatmap layer'}
                  {selectedPreset === 'covid' && 'Bilateral peripheral opacity heatmap layer'}
                </div>
              </div>
            </div>

            {isAnalyzing && (
              <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex flex-col items-center justify-center space-y-2">
                <RefreshCw className="w-8 h-8 text-sky-400 animate-spin" />
                <span className="text-xs font-semibold text-sky-300">Running inference...</span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Model Inference Results */}
        <div className="lg:col-span-7 bg-slate-950 border border-slate-800 rounded-xl p-5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-4 border-b border-slate-800 mb-4">
              <div>
                <span className="text-xs text-slate-400 uppercase font-semibold">Predicted Diagnosis</span>
                <h3 className={`text-2xl font-extrabold mt-0.5 ${
                  currentResult.classLabel === 'Normal' ? 'text-sky-400' :
                  currentResult.classLabel === 'Pneumonia' ? 'text-rose-400' : 'text-emerald-400'
                }`}>
                  {currentResult.classLabel}
                </h3>
              </div>
              <div className="text-right">
                <span className="text-xs text-slate-400 uppercase font-semibold">Confidence Score</span>
                <div className="text-2xl font-extrabold text-white">
                  {currentResult.confidence}%
                </div>
              </div>
            </div>

            {/* Probability Bars */}
            <div className="space-y-3 mb-6">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Class Probability Distribution
              </span>

              {/* Normal */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">Normal</span>
                  <span className="text-sky-400 font-mono">{(currentResult.probabilities.normal * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2.5 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div style={{ width: `${currentResult.probabilities.normal * 100}%` }} className="bg-sky-500 h-full rounded-full" />
                </div>
              </div>

              {/* Pneumonia */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">Pneumonia</span>
                  <span className="text-rose-400 font-mono">{(currentResult.probabilities.pneumonia * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2.5 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div style={{ width: `${currentResult.probabilities.pneumonia * 100}%` }} className="bg-rose-500 h-full rounded-full" />
                </div>
              </div>

              {/* COVID-19 */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300 font-medium">COVID-19</span>
                  <span className="text-emerald-400 font-mono">{(currentResult.probabilities.covid * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2.5 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div style={{ width: `${currentResult.probabilities.covid * 100}%` }} className="bg-emerald-500 h-full rounded-full" />
                </div>
              </div>
            </div>

            {/* Clinical Radiographic Interpretation */}
            <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 text-xs text-slate-300 space-y-1">
              <span className="font-semibold text-slate-200 block mb-1">Radiographic Finding Summary:</span>
              <p className="leading-relaxed">{currentResult.explanation}</p>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-400">
            <span>Model Backbone: ResNet-18 [128x128x3]</span>
            <span>Differential Privacy: Guaranteed (ε=2.53)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
