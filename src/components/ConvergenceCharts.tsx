import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend, BarChart, Bar } from 'recharts';

export interface RoundRecord {
  round: number;
  trainLoss: number;
  testLoss: number;
  testAccuracy: number;
  privacyEpsilon: number;
  uncompressedMb: number;
  compressedMb: number;
  rocAuc: number;
}

interface ConvergenceChartsProps {
  history: RoundRecord[];
}

export const ConvergenceCharts: React.FC<ConvergenceChartsProps> = ({ history }) => {
  const chartData = history.map((item) => ({
    round: `R${item.round}`,
    Accuracy: parseFloat((item.testAccuracy * 100).toFixed(2)),
    TrainLoss: parseFloat(item.trainLoss.toFixed(4)),
    TestLoss: parseFloat(item.testLoss.toFixed(4)),
    Epsilon: parseFloat(item.privacyEpsilon.toFixed(2)),
    Uncompressed: parseFloat(item.uncompressedMb.toFixed(1)),
    Compressed: parseFloat(item.compressedMb.toFixed(1)),
  }));

  const latest = history[history.length - 1];
  const peakAccuracy = history.reduce((max, r) => Math.max(max, r.testAccuracy), 0);
  const peakRound = history.find((r) => r.testAccuracy === peakAccuracy)?.round ?? latest.round;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
      {/* Chart 1: FL Convergence (Accuracy) */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              FL Test Accuracy Convergence
            </h3>
            <p className="text-xs text-slate-400">
              Peak: {(peakAccuracy * 100).toFixed(2)}% (Round {peakRound}) - from loaded results
            </p>
          </div>
        </div>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="round" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis domain={[40, 100]} stroke="#64748b" tick={{ fontSize: 11 }} unit="%" />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc', borderRadius: '8px' }}
                formatter={(val: any) => [`${val}%`, 'Test Accuracy']}
              />
              <Legend />
              <Line type="monotone" dataKey="Accuracy" stroke="#0284c7" strokeWidth={3} dot={{ r: 3, fill: '#38bdf8' }} activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2: Loss Trajectory */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Cross-Entropy Loss Trajectory
            </h3>
            <p className="text-xs text-slate-400">Hospital Train Loss vs Global Test Loss</p>
          </div>
          <span className="text-xs font-semibold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 px-2.5 py-1 rounded-md">
            FedProx (μ=0.01)
          </span>
        </div>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="round" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc', borderRadius: '8px' }}
              />
              <Legend />
              <Line type="monotone" dataKey="TrainLoss" stroke="#ef4444" strokeWidth={2} name="Avg Train Loss" dot={false} />
              <Line type="monotone" dataKey="TestLoss" stroke="#3b82f6" strokeWidth={2} strokeDasharray="4 4" name="Global Test Loss" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 3: DP Epsilon Expenditure */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              RDP Privacy Epsilon (ε) Budget
            </h3>
            <p className="text-xs text-slate-400">Max privacy guarantee budget threshold ε = 3.0</p>
          </div>
          <span className="text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2.5 py-1 rounded-md">
            δ = 10⁻⁴
          </span>
        </div>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="round" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 4]} stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc', borderRadius: '8px' }}
                formatter={(val: any) => [`ε = ${val}`, 'Privacy Budget']}
              />
              <Legend />
              <Line type="monotone" dataKey="Epsilon" stroke="#10b981" strokeWidth={3} name="Cumulative Epsilon (ε)" dot={{ r: 2 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 4: Communication Data Volume */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">
              Cumulative Communication Payload (MB)
            </h3>
            <p className="text-xs text-slate-400">Uncompressed FL vs Top-K Sparse FL (68% Reduction)</p>
          </div>
          <span className="text-xs font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20 px-2.5 py-1 rounded-md">
            Top-K (20%)
          </span>
        </div>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData.filter((_, idx) => idx % 5 === 0 || idx === chartData.length - 1)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="round" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} unit="MB" />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc', borderRadius: '8px' }}
                formatter={(val: any) => [`${val} MB`]}
              />
              <Legend />
              <Bar dataKey="Uncompressed" fill="#475569" name="Uncompressed FL" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Compressed" fill="#0284c7" name="Proposed Top-K FL" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
