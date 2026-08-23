import React, { useEffect, useState, useRef } from 'react';
import { FileText, UploadCloud, RefreshCw } from 'lucide-react';

// @ts-ignore
import defaultReportMarkdown from '../../reports/final_experiment_report.md?raw';

export const ReportViewer: React.FC = () => {
  const [reportText, setReportText] = useState<string | null>(defaultReportMarkdown);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchReport = async () => {
    setLoading(true);
    try {
      const res = await fetch('/reports/final_experiment_report.md');
      if (res.ok) {
        const text = await res.text();
        setReportText(text);
      } else {
        // Fallback to the imported report
        setReportText(defaultReportMarkdown);
      }
    } catch (e) {
      setReportText(defaultReportMarkdown);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReport();
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      setReportText(event.target?.result as string);
    };
    reader.readAsText(file);
  };

  return (
    <div className="space-y-6 mb-6">
      {/* Generated Report Markdown Preview */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-4 flex-wrap gap-3">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <FileText className="w-5 h-5 text-emerald-400" />
              Final Major Research Project Report (Generated)
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">Location: <code className="text-slate-300">reports/final_experiment_report.md</code></p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={fetchReport}
              className="flex items-center gap-1 px-3 py-1.5 rounded-md text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Reload From Server
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".md"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-emerald-800 hover:bg-emerald-700 text-emerald-100 border border-emerald-700"
            >
              <UploadCloud className="w-3.5 h-3.5" />
              Upload report.md
            </button>
          </div>
        </div>

        <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 text-xs text-slate-300 font-mono leading-relaxed overflow-x-auto whitespace-pre-wrap min-h-[300px]">
          {loading ? (
            <div className="text-slate-500 text-center py-10">Fetching latest report...</div>
          ) : reportText ? (
            reportText
          ) : (
            <div className="text-slate-500 text-center py-10">No report loaded. Use the controls above to load your report.</div>
          )}
        </div>
      </div>
    </div>
  );
};
