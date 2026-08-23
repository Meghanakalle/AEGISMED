import React, { useState } from 'react';
import { Building2, Server } from 'lucide-react';
import { parseCsv, readFileAsText } from '../lib/csv';

// @ts-ignore
import defaultHospitalStatsCsv from '../../results/hospital_stats.csv?raw';

interface HospitalRow {
  hospitalId: string;
  name: string;
  sampleCount: number;
  classCounts: Record<string, number>;
}

const BAR_COLORS = [
  { bar: 'bg-sky-500', chip: 'bg-sky-500/10 border-sky-500/20 text-sky-300' },
  { bar: 'bg-rose-500', chip: 'bg-rose-500/10 border-rose-500/20 text-rose-300' },
  { bar: 'bg-emerald-500', chip: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300' },
  { bar: 'bg-amber-500', chip: 'bg-amber-500/10 border-amber-500/20 text-amber-300' },
  { bar: 'bg-violet-500', chip: 'bg-violet-500/10 border-violet-500/20 text-violet-300' },
];

const NON_CLASS_COLUMNS = new Set(['hospital_id', 'hospital_name', 'sample_count']);

function parseHospitalData(text: string) {
  try {
    const rows = parseCsv(text);
    if (rows.length === 0) return null;
    const cols = Object.keys(rows[0]).filter((c) => !NON_CLASS_COLUMNS.has(c));
    const parsed: HospitalRow[] = rows.map((r) => {
      const classCounts: Record<string, number> = {};
      cols.forEach((c) => {
        classCounts[c] = Number(r[c]) || 0;
      });
      return {
        hospitalId: r['hospital_id'],
        name: r['hospital_name'],
        sampleCount: Number(r['sample_count']) || 0,
        classCounts,
      };
    });
    return { classNames: cols, hospitals: parsed };
  } catch (e) {
    return null;
  }
}

const initialHospitalData = parseHospitalData(defaultHospitalStatsCsv);

export const HospitalNetwork: React.FC = () => {
  const [hospitals, setHospitals] = useState<HospitalRow[] | null>(initialHospitalData?.hospitals ?? null);
  const [classNames, setClassNames] = useState<string[]>(initialHospitalData?.classNames ?? []);

  return (
    <div className="space-y-6 mb-6">
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Building2 className="w-5 h-5 text-sky-400" />
              Hospital Network Topology
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Real per-hospital sample counts and class distribution from your Dirichlet partition.
            </p>
          </div>
        </div>

        {!hospitals ? (
          <div className="text-xs text-slate-500 border border-dashed border-slate-700 rounded-lg p-6 text-center">
            No hospital data loaded yet. Run any training config (results/hospital_stats.csv is written
            automatically alongside experiment_metrics.csv), then load it here.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {hospitals.map((h) => (
              <div key={h.hospitalId} className="bg-slate-950/80 border border-slate-800 rounded-xl p-5 hover:border-slate-700 transition-all">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className="p-2.5 bg-sky-500/10 border border-sky-500/20 rounded-lg text-sky-400">
                      <Server className="w-5 h-5" />
                    </div>
                    <h4 className="font-semibold text-white text-sm">{h.name}</h4>
                  </div>
                  <span className="px-2.5 py-0.5 bg-sky-500/10 text-sky-400 border border-sky-500/20 rounded-full text-xs font-medium">
                    {h.sampleCount} Samples
                  </span>
                </div>

                <div className="space-y-2 mt-4">
                  <div className="h-3 w-full bg-slate-800 rounded-full overflow-hidden flex">
                    {classNames.map((cls, idx) => {
                      const pct = h.sampleCount > 0 ? (h.classCounts[cls] / h.sampleCount) * 100 : 0;
                      const color = BAR_COLORS[idx % BAR_COLORS.length];
                      return (
                        <div
                          key={cls}
                          style={{ width: `${pct}%` }}
                          className={`${color.bar} h-full`}
                          title={`${cls}: ${h.classCounts[cls]} (${pct.toFixed(1)}%)`}
                        />
                      );
                    })}
                  </div>

                  <div className="grid gap-2 pt-2 text-center text-xs" style={{ gridTemplateColumns: `repeat(${classNames.length}, minmax(0, 1fr))` }}>
                    {classNames.map((cls, idx) => {
                      const pct = h.sampleCount > 0 ? ((h.classCounts[cls] / h.sampleCount) * 100).toFixed(1) : '0.0';
                      const color = BAR_COLORS[idx % BAR_COLORS.length];
                      return (
                        <div key={cls} className={`${color.chip} border rounded py-1 px-1`}>
                          {cls}: <span className="font-bold">{h.classCounts[cls]}</span> ({pct}%)
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
