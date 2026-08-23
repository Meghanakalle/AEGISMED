import React, { useState } from 'react';
import { BarChart3, Layers, Sliders } from 'lucide-react';
import { parseCsv, readFileAsText } from '../lib/csv';

// @ts-ignore
import defaultAblationCsv from '../../results/ablation_study_summary.csv?raw';
// @ts-ignore
import defaultAlphaCsv from '../../results/non_iid_alpha_study.csv?raw';

// Columns written by experiments/ablation.py -> results/ablation_study_summary.csv
interface AblationRow {
  variant: string;
  accuracyPct: number;
  rocAuc: number;
  epsilon: string;
  commReductionPct: number;
}

// Columns written by experiments/non_iid_study.py -> results/non_iid_alpha_study.csv
interface AlphaRow {
  alpha: string;
  level: string;
  accuracyPct: number;
  loss: number;
  rocAuc: number;
}

function parseAblationData(text: string): AblationRow[] | null {
  try {
    const rows = parseCsv(text);
    const parsed = rows.map((r) => ({
      variant: r['Variant'],
      accuracyPct: Number(r['Test Accuracy (%)']),
      rocAuc: Number(r['ROC-AUC']),
      epsilon: r['Epsilon (ε)'] || r['Epsilon'] || '0.0',
      commReductionPct: Number(r['Comm Reduction (%)'] || r['Comm Reduction (%)']),
    }));
    if (parsed.length === 0 || parsed.some((p) => Number.isNaN(p.accuracyPct))) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

function parseAlphaData(text: string): AlphaRow[] | null {
  try {
    const rows = parseCsv(text);
    const parsed = rows.map((r) => ({
      alpha: r['Dirichlet Alpha (α)'] || r['Dirichlet Alpha (alpha)'] || 'unknown',
      level: r['Heterogeneity Level'],
      accuracyPct: Number(r['Final Test Acc (%)']),
      loss: Number(r['Final Loss']),
      rocAuc: Number(r['ROC-AUC']),
    }));
    if (parsed.length === 0 || parsed.some((p) => Number.isNaN(p.accuracyPct))) {
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

const initialAblation = parseAblationData(defaultAblationCsv);
const initialAlpha = parseAlphaData(defaultAlphaCsv);

export const AblationViewer: React.FC = () => {
  const [ablationRows, setAblationRows] = useState<AblationRow[] | null>(initialAblation);
  const [alphaRows, setAlphaRows] = useState<AlphaRow[] | null>(initialAlpha);

  return (
    <div className="space-y-6 mb-6">
      {/* Ablation Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Layers className="w-5 h-5 text-sky-400" />
              5-Variant Experimental Ablation Study
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Evaluating individual component contributions: FedProx proximal term, Differential Privacy, and Top-K compression.
            </p>
          </div>
        </div>

        {!ablationRows ? (
          <div className="text-xs text-slate-500 border border-dashed border-slate-700 rounded-lg p-6 text-center">
            No ablation results loaded.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider">
                  <th className="py-3 px-3">Architecture Variant</th>
                  <th className="py-3 px-3">Test Acc (%)</th>
                  <th className="py-3 px-3">ROC-AUC</th>
                  <th className="py-3 px-3">Epsilon (ε)</th>
                  <th className="py-3 px-3">Comm Savings</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 text-slate-300">
                {ablationRows.map((item, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/50 transition-colors">
                    <td className="py-3 px-3 font-semibold">{item.variant}</td>
                    <td className="py-3 px-3 font-bold text-sky-400">{item.accuracyPct.toFixed(2)}%</td>
                    <td className="py-3 px-3 font-mono">{item.rocAuc.toFixed(3)}</td>
                    <td className="py-3 px-3 font-mono">{item.epsilon}</td>
                    <td className="py-3 px-3 font-mono text-emerald-400">{item.commReductionPct.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Non-IID Dirichlet Sensitivity Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Sliders className="w-5 h-5 text-indigo-400" />
              Non-IID Statistical Heterogeneity Dirichlet Sensitivity (α)
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Impact of statistical data skew across hospitals on global accuracy and loss convergence.
            </p>
          </div>
        </div>

        {!alphaRows ? (
          <div className="text-xs text-slate-500 border border-dashed border-slate-700 rounded-lg p-6 text-center">
            No non-IID sensitivity results loaded yet.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 font-semibold uppercase tracking-wider">
                  <th className="py-3 px-3">Dirichlet Alpha (α)</th>
                  <th className="py-3 px-3">Heterogeneity Level</th>
                  <th className="py-3 px-3">Test Accuracy (%)</th>
                  <th className="py-3 px-3">Test Loss</th>
                  <th className="py-3 px-3">Macro ROC-AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 text-slate-300">
                {alphaRows.map((item, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/50 transition-colors">
                    <td className="py-3 px-3 font-bold font-mono">α = {item.alpha}</td>
                    <td className="py-3 px-3">{item.level}</td>
                    <td className="py-3 px-3 font-bold text-indigo-400">{item.accuracyPct.toFixed(2)}%</td>
                    <td className="py-3 px-3 font-mono">{item.loss.toFixed(4)}</td>
                    <td className="py-3 px-3 font-mono">{item.rocAuc.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
