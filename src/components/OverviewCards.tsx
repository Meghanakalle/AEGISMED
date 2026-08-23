import React from 'react';
import { Activity, ShieldCheck, ArrowDownRight, Network } from 'lucide-react';

interface MetricsData {
  round: number;
  accuracy: number;
  bestAccuracy: number;
  bestRound: number;
  epsilon: number;
  commSavings: number;
  uncompressedMb: number;
  compressedMb: number;
  rocAuc: number;
}

export const OverviewCards: React.FC<{ metrics: MetricsData }> = ({ metrics }) => {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
      {/* Accuracy Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm relative overflow-hidden">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Global Test Accuracy
          </span>
          <div className="p-2 bg-sky-500/10 border border-sky-500/20 rounded-lg text-sky-400">
            <Activity className="w-5 h-5" />
          </div>
        </div>
        <div className="flex items-baseline space-x-2">
          <span className="text-3xl font-extrabold text-white">{(metrics.accuracy * 100).toFixed(2)}%</span>
          <span className="text-xs font-medium text-slate-400 bg-slate-800 border border-slate-700 px-2 py-0.5 rounded-full">
            Round {metrics.round} (final)
          </span>
        </div>
        {metrics.bestRound !== metrics.round && (
          <p className="text-xs text-emerald-400 mt-2 font-medium">
            Best: {(metrics.bestAccuracy * 100).toFixed(2)}% at Round {metrics.bestRound} - saved to best_model.pt
          </p>
        )}
        <p className="text-xs text-slate-400 mt-2">
          ROC-AUC: <span className="text-slate-200 font-semibold">{metrics.rocAuc.toFixed(4)}</span> (from loaded results)
        </p>
      </div>

      {/* Privacy Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm relative overflow-hidden">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            DP Privacy Guarantee
          </span>
          <div className="p-2 bg-emerald-500/10 border border-emerald-500/20 rounded-lg text-emerald-400">
            <ShieldCheck className="w-5 h-5" />
          </div>
        </div>
        <div className="flex items-baseline space-x-2">
          <span className="text-3xl font-extrabold text-white">ε = {metrics.epsilon.toFixed(2)}</span>
          <span className="text-xs font-medium text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full">
            δ = 10⁻⁴
          </span>
        </div>
        <p className="text-xs text-slate-400 mt-2">
          Gaussian noise <span className="text-slate-200 font-semibold">σ = 1.1</span> | Max grad norm <span className="text-slate-200 font-semibold">C = 1.0</span>
        </p>
      </div>

      {/* Bandwidth Savings Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm relative overflow-hidden">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Bandwidth Overhead
          </span>
          <div className="p-2 bg-indigo-500/10 border border-indigo-500/20 rounded-lg text-indigo-400">
            <ArrowDownRight className="w-5 h-5" />
          </div>
        </div>
        <div className="flex items-baseline space-x-2">
          <span className="text-3xl font-extrabold text-white">-{metrics.commSavings.toFixed(1)}%</span>
          <span className="text-xs font-medium text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded-full">
            Top-K (20%)
          </span>
        </div>
        <p className="text-xs text-slate-400 mt-2">
          Transmitted: <span className="text-slate-200 font-semibold">{metrics.compressedMb.toFixed(0)} MB</span> (Saved {(metrics.uncompressedMb - metrics.compressedMb).toFixed(0)} MB)
        </p>
      </div>

      {/* Hospital Network Card */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-sm relative overflow-hidden">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Heterogeneous Nodes
          </span>
          <div className="p-2 bg-amber-500/10 border border-amber-500/20 rounded-lg text-amber-400">
            <Network className="w-5 h-5" />
          </div>
        </div>
        <div className="flex items-baseline space-x-2">
          <span className="text-3xl font-extrabold text-white">4 Hospitals</span>
          <span className="text-xs font-medium text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded-full">
            Dirichlet α=0.3
          </span>
        </div>
        <p className="text-xs text-slate-400 mt-2">
          FedProx regularization <span className="text-slate-200 font-semibold">μ = 0.01</span> against statistical drift
        </p>
      </div>
    </div>
  );
};
