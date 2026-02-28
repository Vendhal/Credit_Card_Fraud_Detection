import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart3, 
  TrendingUp, 
  ShieldAlert, 
  CreditCard,
  Download,
  RefreshCw
} from 'lucide-react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
  BarChart, Bar
} from 'recharts';
import { formatCurrency } from '../lib/utils';
import { MOCK_TRANSACTIONS } from '../lib/mockData';

const COLORS = ['#16a34a', '#dc2626'];

interface BOCResult {
  filename: string;
  test_f1: number;
  test_precision: number;
  test_recall: number;
  confusion_matrix: {
    tn: number;
    fp: number;
    fn: number;
    tp: number;
  };
  threshold: number;
  fold_results: Array<{ fold: number; f1: number }>;
  synthetic: number;
  time_minutes: number;
  timestamp: string;
}

interface TrainingLog {
  filename: string;
  rounds: number;
  f1_scores: number[];
  best_f1: number;
  timestamp: string;
}

const StatCard = ({ title, value, icon: Icon, subtext, delay }: any) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay }}
    className="bg-card p-6 rounded-2xl shadow-sm border border-border"
  >
    <div className="flex items-center justify-between mb-4">
      <div className="p-3 bg-secondary rounded-xl text-primary">
        <Icon size={24} />
      </div>
    </div>
    <p className="text-muted-foreground text-sm font-medium mb-1">{title}</p>
    <h3 className="text-3xl font-bold tracking-tight">{value}</h3>
    {subtext && <p className="text-xs text-muted-foreground mt-1">{subtext}</p>}
  </motion.div>
);

export default function Dashboard() {
  const [results, setResults] = useState<BOCResult[]>([]);
  const [logs, setLogs] = useState<TrainingLog[]>([]);
  const [selectedResult, setSelectedResult] = useState<BOCResult | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [resultsRes, logsRes] = await Promise.all([
        fetch('http://localhost:5000/api/results'),
        fetch('http://localhost:5000/api/logs')
      ]);
      
      const resultsData = await resultsRes.json();
      const logsData = await logsRes.json();
      
      if (Array.isArray(resultsData) && resultsData.length > 0) {
        setResults(resultsData);
        setSelectedResult(resultsData[0]);
      }
      
      if (Array.isArray(logsData)) {
        setLogs(logsData);
      }
    } catch (err) {
      console.error('Failed to fetch BOC data:', err);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
  }, []);

  const currentResult = selectedResult || results[0];
  
  const foldData = currentResult?.fold_results || [];
  const cm = currentResult?.confusion_matrix || { tn: 0, fp: 0, fn: 0, tp: 0 };
  
  const distributionData = [
    { name: 'Legitimate', value: cm.tn + cm.fp },
    { name: 'Fraudulent', value: cm.tp + cm.fn },
  ];

  const trainingData = logs[0]?.f1_scores.map((f1, i) => ({
    round: i + 1,
    f1: f1
  })) || [];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="animate-spin h-8 w-8" />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">BOC Dashboard</h1>
          <p className="text-muted-foreground mt-1">Battle of Civilizations - Fraud Detection Results</p>
        </div>
        <div className="flex gap-2">
          {results.length > 1 && (
            <select 
              className="px-3 py-2 bg-secondary rounded-lg text-sm"
              onChange={(e) => {
                const idx = parseInt(e.target.value);
                setSelectedResult(results[idx]);
              }}
              value={results.findIndex(r => r.filename === selectedResult?.filename) || 0}
            >
              {results.map((r, i) => (
                <option key={i} value={i}>
                  {r.filename.replace('ultimate_safe_oof_', '').replace('.json', '')}
                </option>
              ))}
            </select>
          )}
          <button 
            onClick={fetchData}
            className="flex items-center gap-2 px-4 py-2 bg-secondary text-secondary-foreground hover:bg-secondary/80 rounded-lg font-medium transition-colors"
          >
            <RefreshCw size={18} />
            Refresh
          </button>
        </div>
      </div>

      {currentResult && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <StatCard 
              title="Test F1 Score" 
              value={(currentResult.test_f1 * 100).toFixed(2) + '%'} 
              icon={ShieldAlert} 
              subtext="Top 10% on Kaggle"
              delay={0}
            />
            <StatCard 
              title="Precision" 
              value={(currentResult.test_precision * 100).toFixed(2) + '%'} 
              icon={BarChart3} 
              subtext={`False alarms: ${cm.fp}`}
              delay={0.1}
            />
            <StatCard 
              title="Recall" 
              value={(currentResult.test_recall * 100).toFixed(2) + '%'} 
              icon={TrendingUp} 
              subtext={`Caught ${cm.tp}/${cm.tp + cm.fn} frauds`}
              delay={0.2}
            />
            <StatCard 
              title="Threshold" 
              value={currentResult.threshold?.toFixed(4) || 'N/A'} 
              icon={CreditCard} 
              subtext={`${currentResult.synthetic} synthetic samples`}
              delay={0.3}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Fold Results */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 }}
              className="bg-card p-6 rounded-2xl shadow-sm border border-border"
            >
              <h3 className="text-lg font-semibold mb-6">OOF Fold F1 Scores</h3>
              <div className="h-[300px] w-full">
                {foldData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={foldData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                      <XAxis dataKey="fold" stroke="#888888" fontSize={12} />
                      <YAxis stroke="#888888" fontSize={12} domain={[0.8, 1]} />
                      <RechartsTooltip />
                      <Bar dataKey="f1" fill="#6366f1" radius={[4, 4, 0, 0]} name="F1" />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-muted-foreground">
                    No fold data
                  </div>
                )}
              </div>
            </motion.div>

            {/* Confusion Matrix / Distribution */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5 }}
              className="bg-card p-6 rounded-2xl shadow-sm border border-border flex flex-col"
            >
              <h3 className="text-lg font-semibold mb-6">Prediction Distribution</h3>
              <div className="flex-1 min-h-[300px] relative">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={distributionData}
                      cx="50%"
                      cy="50%"
                      innerRadius={80}
                      outerRadius={110}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {distributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                    <Legend verticalAlign="bottom" height={36}/>
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </motion.div>

            {/* Training Progress */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 }}
              className="bg-card p-6 rounded-2xl shadow-sm border border-border"
            >
              <h3 className="text-lg font-semibold mb-6">Training Progress</h3>
              <div className="h-[300px] w-full">
                {trainingData.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={trainingData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="colorF1" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="round" stroke="#888888" fontSize={12} />
                      <YAxis stroke="#888888" fontSize={12} domain={[0.8, 1]} />
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                      <RechartsTooltip />
                      <Area type="monotone" dataKey="f1" stroke="#6366f1" fillOpacity={1} fill="url(#colorF1)" name="F1" />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-muted-foreground">
                    No training data
                  </div>
                )}
              </div>
            </motion.div>
          </div>

          {/* Stats Row */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center"
          >
            <div className="bg-muted/50 p-4 rounded-xl">
              <p className="text-sm text-muted-foreground">True Negatives</p>
              <p className="text-2xl font-bold text-safe">{cm.tn.toLocaleString()}</p>
            </div>
            <div className="bg-muted/50 p-4 rounded-xl">
              <p className="text-sm text-muted-foreground">False Positives</p>
              <p className="text-2xl font-bold text-destructive">{cm.fp}</p>
            </div>
            <div className="bg-muted/50 p-4 rounded-xl">
              <p className="text-sm text-muted-foreground">False Negatives</p>
              <p className="text-2xl font-bold text-destructive">{cm.fn}</p>
            </div>
            <div className="bg-muted/50 p-4 rounded-xl">
              <p className="text-sm text-muted-foreground">True Positives</p>
              <p className="text-2xl font-bold text-safe">{cm.tp}</p>
            </div>
          </motion.div>
        </>
      )}

      {!currentResult && (
        <div className="text-center py-12 text-muted-foreground">
          <p>No BOC results found. Run the pipeline first!</p>
          <p className="text-sm mt-2">python ultimate_hive_convergence_safe_oof.py</p>
        </div>
      )}
    </div>
  );
}
