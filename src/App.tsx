import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { OverviewCards } from './components/OverviewCards';
import { ConvergenceCharts, RoundRecord } from './components/ConvergenceCharts';
import { HospitalNetwork } from './components/HospitalNetwork';
import { DiagnosticPlayground } from './components/DiagnosticPlayground';
import { AblationViewer } from './components/AblationViewer';
import { parseCsv, readFileAsText } from './lib/csv';
import { FileWarning, UploadCloud } from 'lucide-react';

// @ts-ignore
import defaultMetricsCsv from '../results/experiment_metrics.csv?raw';

// Maps the columns written by federated/server.py's
// df_results.to_csv("results/experiment_metrics.csv") into RoundRecord.
function rowsToHistory(rows: Record<string, string>[]): RoundRecord[] {
  return rows
    .map((row) => ({
      round: Number(row.round),
      trainLoss: Number(row.train_loss),
      testLoss: Number(row.test_loss),
      testAccuracy: Number(row.test_accuracy),
      privacyEpsilon: Number(row.privacy_epsilon),
      uncompressedMb: Number(row.uncompressed_comm_mb),
      compressedMb: Number(row.compressed_comm_mb),
      rocAuc: Number(row.roc_auc),
    }))
    .filter((r) => Number.isFinite(r.round))
    .sort((a, b) => a.round - b.round);
}

const initialHistory = rowsToHistory(parseCsv(defaultMetricsCsv));

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [history, setHistory] = useState<RoundRecord[]>(initialHistory);
  const [sourceFileName, setSourceFileName] = useState<string | null>(
    initialHistory.length > 0 ? 'results/experiment_metrics.csv' : null
  );
  const [loadError, setLoadError] = useState<string | null>(null);

  const handleLoadCsv = async (file: File) => {
    setLoadError(null);
    try {
      const text = await readFileAsText(file);
      const rows = parseCsv(text);
      const parsed = rowsToHistory(rows);
      if (parsed.length === 0) {
        setLoadError(
          `Couldn't find round/test_accuracy columns in "${file.name}". Load the results/experiment_metrics.csv produced by run_experiment.py.`
        );
        return;
      }
      setHistory(parsed);
      setSourceFileName(file.name);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Failed to read file.');
    }
  };

  const hasData = history.length > 0;
  const latest = hasData ? history[history.length - 1] : null;
  // The checkpoint saved to best_model.pt is the BEST round by accuracy, not
  // necessarily the last round (federated/server.py only overwrites it when
  // accuracy improves). Surface that here instead of just the last row, since
  // late-round  noise on small datasets can make the final round look worse
  // than the model you actually end up with.
  const best = hasData
    ? history.reduce((a, b) => (b.testAccuracy > a.testAccuracy ? b : a), history[0])
    : null;

  const overviewMetrics = latest
    ? {
        round: latest.round,
        accuracy: latest.testAccuracy,
        bestAccuracy: best ? best.testAccuracy : latest.testAccuracy,
        bestRound: best ? best.round : latest.round,
        epsilon: latest.privacyEpsilon,
        commSavings:
          latest.uncompressedMb > 0
            ? ((latest.uncompressedMb - latest.compressedMb) / latest.uncompressedMb) * 100
            : 0,
        uncompressedMb: latest.uncompressedMb,
        compressedMb: latest.compressedMb,
        rocAuc: latest.rocAuc,
      }
    : null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {!hasData ? (
          <div className="bg-slate-900 border border-dashed border-slate-700 rounded-xl p-10 text-center mb-6">
            <UploadCloud className="w-10 h-10 mx-auto text-slate-500 mb-3" />
            <h2 className="text-white font-semibold mb-2">No results loaded yet</h2>
            <p className="text-sm text-slate-400 max-w-lg mx-auto mb-4">
              This dashboard only shows real numbers from an actual training run - it no longer
              generates simulated data. Run an experiment first.
            </p>
            <code className="block bg-slate-950 border border-slate-800 rounded-lg px-4 py-2 text-xs text-sky-400 mb-4 max-w-md mx-auto">
              python run_experiment.py --config configs/covid_radiography_research.yaml --mode proposed
            </code>
            {loadError && (
              <p className="text-xs text-rose-400 flex items-center justify-center gap-1.5 mt-2">
                <FileWarning className="w-3.5 h-3.5" /> {loadError}
              </p>
            )}
          </div>
        ) : (
          <>
            {loadError && (
              <p className="text-xs text-rose-400 flex items-center gap-1.5 mb-4">
                <FileWarning className="w-3.5 h-3.5" /> {loadError}
              </p>
            )}
            <p className="text-xs text-slate-500 mb-4">
              Showing real results loaded from{' '}
              <span className="text-slate-300 font-medium">{sourceFileName}</span> - {history.length}{' '}
              round{history.length === 1 ? '' : 's'}.
            </p>

            {overviewMetrics && <OverviewCards metrics={overviewMetrics} />}

            {activeTab === 'dashboard' && <ConvergenceCharts history={history} />}
            {activeTab === 'hospitals' && <HospitalNetwork />}
            {activeTab === 'diagnostics' && <DiagnosticPlayground />}
            {activeTab === 'ablation' && <AblationViewer />}
          </>
        )}
      </main>
    </div>
  );
}
