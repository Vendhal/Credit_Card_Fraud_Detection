import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import * as z from 'zod';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Bot, 
  MapPin, 
  CreditCard, 
  Store, 
  DollarSign, 
  Clock,
  ShieldAlert,
  ShieldCheck,
  Zap
} from 'lucide-react';
import { formatCurrency } from '../lib/utils';
import { toast } from 'react-toastify';

const schema = z.object({
  amount: z.number().positive("Amount must be positive"),
  time: z.string().min(1, "Time is required"),
  location: z.string().min(2, "Location is required"),
  merchantCategory: z.string().min(1, "Category is required"),
  cardType: z.string().min(1, "Card type is required"),
  cardNumber: z.string().min(12, "Card number is too short"),
  cardHolder: z.string().min(1, "Cardholder name is required"),
});

type FormData = z.infer<typeof schema>;

interface PredictionResult {
  isFraud: boolean;
  score: number;
  factors: string[];
  model?: string;
}

function luhnCheck(cardNumber: string) {
  const digits = cardNumber.replace(/\D/g, '');
  let sum = 0;
  let alt = false;
  for (let i = digits.length - 1; i >= 0; i--) {
    let n = parseInt(digits.charAt(i), 10);
    if (alt) {
      n = n * 2;
      if (n > 9) n -= 9;
    }
    sum += n;
    alt = !alt;
  }
  return (sum % 10) === 0;
}

function detectCardType(cardNumber: string) {
  const digits = cardNumber.replace(/\D/g, '');
  if (digits.startsWith('4')) return 'visa';
  if (/^5[1-5]/.test(digits) || /^2(2[2-9]|[3-6]\d|7[01])/.test(digits) || /^2720/.test(digits)) return 'mastercard';
  if (digits.startsWith('34') || digits.startsWith('37')) return 'amex';
  if (digits.startsWith('60') || digits.startsWith('62') || digits.startsWith('64') || digits.startsWith('65')) return 'discover';
  return 'unknown';
}

export default function Predict() {
  const [isPredicting, setIsPredicting] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [usingRealModel, setUsingRealModel] = useState<boolean | null>(null);

  // Check model status on load
  useState(() => {
    fetch('http://localhost:5000/api/model-info')
      .then(res => res.json())
      .then(data => {
        setUsingRealModel(data.using_real_model ?? false);
      })
      .catch(() => setUsingRealModel(false));
  });

  const { register, handleSubmit, formState: { errors } } = useForm<FormData>({
    resolver: zodResolver(schema),
    defaultValues: {
      time: new Date().toISOString().slice(0, 16),
      cardHolder: '',
      cardNumber: ''
    }
  });

  const onSubmit = async (data: FormData) => {
    setIsPredicting(true);
    setResult(null);

    try {
      // Card validation checks
      const rawCard = data.cardNumber.replace(/\s+/g, '');
      const isLuhnValid = luhnCheck(rawCard);
      const detectedType = detectCardType(rawCard);

      // Send transaction details to API - it will generate proper V1-V28 features
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          amount: data.amount,
          merchantCategory: data.merchantCategory,
          cardType: data.cardType
        })
      });

      let apiResult;
      if (response.ok) {
        apiResult = await response.json();
      } else {
        // Fallback if API not available
        apiResult = null;
      }

      let factors: string[] = [];
      let score = 0;
      let isFraud = false;

      if (apiResult && apiResult.score !== undefined) {
        // Use API result
        score = apiResult.score;
        isFraud = apiResult.isFraud;
        factors.push(`Model: ${apiResult.model || 'Unknown'}`);
        factors.push(`Score: ${score.toFixed(4)}, Threshold: ${apiResult.threshold?.toFixed(4)}`);
        factors.push(`Model: ${apiResult.model || 'BOC-GreatConvergence'}`);
      } else {
        // Fallback to local heuristics
        if (!isLuhnValid) {
          score = 0.98;
          factors.push('Invalid card number (Luhn check failed)');
        } else {
          const scoreBase = data.amount > 5000 ? 0.55 : 0.05;
          const categoryModifier = ['crypto', 'electronics', 'gambling'].includes(data.merchantCategory.toLowerCase()) ? 0.18 : 0;
          score = Math.min(0.99, scoreBase + categoryModifier + Math.random() * 0.1);
          factors.push(data.amount > 5000 ? 'High transaction amount' : 'Standard transaction amount');
          if (categoryModifier > 0) factors.push('High-risk merchant category');
        }
        isFraud = score > 0.65;
      }

      // Card type check
      if (detectedType !== 'unknown' && detectedType !== data.cardType.toLowerCase()) {
        factors.push(`Card mismatch (detected: ${detectedType.toUpperCase()})`);
      } else if (isLuhnValid) {
        factors.push(`Valid card (${detectedType.toUpperCase()})`);
      }

      setResult({ isFraud, score, factors });

      if (isFraud) {
        toast.error("Fraudulent transaction detected!");
      } else {
        toast.success("Transaction looks legitimate.");
      }
    } catch (err) {
      console.error('Prediction error:', err);
      toast.error("API not available. Using local validation.");
      
      // Fallback to local
      const rawCard = data.cardNumber.replace(/\s+/g, '');
      const isLuhnValid = luhnCheck(rawCard);
      let score = isLuhnValid ? 0.1 : 0.95;
      if (data.amount > 5000) score += 0.3;
      
      const isFraud = score > 0.65;
      setResult({
        isFraud,
        score,
        factors: ['Local fallback mode']
      });
      
      if (isFraud) toast.error("Fraud detected!");
      else toast.success("Legitimate (local)");
    }

    setIsPredicting(false);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Manual Evaluation</h1>
        <p className="text-muted-foreground mt-1">
          Run a real-time prediction using the trained machine learning model or perform quick card checks locally.
        </p>
        <div className="mt-4 p-4 bg-primary/5 border border-primary/20 rounded-lg flex items-start gap-3 text-sm">
          <Bot className="text-primary mt-0.5 shrink-0" size={18} />
          <p className="text-muted-foreground">
            <strong className="text-foreground">Note:</strong> {usingRealModel 
              ? '🤖 Using REAL trained BOC model for predictions!' 
              : 'Using simple heuristic (train model to use real AI)'}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Form Section */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="bg-card p-6 md:p-8 rounded-2xl shadow-sm border border-border"
        >
          <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
            <Zap size={20} className="text-primary" />
            Transaction & Card Details
          </h2>
          
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-5">
            <div className="space-y-2">
              <label className="text-sm font-medium">Transaction Amount ($)</label>
              <div className="relative">
                <DollarSign className="absolute left-3 top-3 text-muted-foreground" size={18} />
                <input 
                  type="number" 
                  step="0.01"
                  {...register("amount", { valueAsNumber: true })}
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-border bg-background focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all"
                  placeholder="0.00"
                />
              </div>
              {errors.amount && <p className="text-destructive text-xs">{errors.amount.message}</p>}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Date & Time</label>
              <div className="relative">
                <Clock className="absolute left-3 top-3 text-muted-foreground" size={18} />
                <input 
                  type="datetime-local" 
                  {...register("time")}
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-border bg-background focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all"
                />
              </div>
              {errors.time && <p className="text-destructive text-xs">{errors.time.message}</p>}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Location</label>
              <div className="relative">
                <MapPin className="absolute left-3 top-3 text-muted-foreground" size={18} />
                <input 
                  type="text" 
                  {...register("location")}
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-border bg-background focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all"
                  placeholder="City, Country"
                />
              </div>
              {errors.location && <p className="text-destructive text-xs">{errors.location.message}</p>}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Merchant Category</label>
                <div className="relative">
                  <Store className="absolute left-3 top-3 text-muted-foreground" size={18} />
                  <select 
                    {...register("merchantCategory")}
                    className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-border bg-background focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all appearance-none"
                  >
                    <option value="">Select...</option>
                    <option value="retail">Retail</option>
                    <option value="travel">Travel</option>
                    <option value="electronics">Electronics</option>
                    <option value="crypto">Cryptocurrency</option>
                    <option value="food">Food & Dining</option>
                    <option value="services">Services</option>
                    <option value="gambling">Gambling</option>
                  </select>
                </div>
                {errors.merchantCategory && <p className="text-destructive text-xs">{errors.merchantCategory.message}</p>}
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Card Type</label>
                <div className="relative">
                  <CreditCard className="absolute left-3 top-3 text-muted-foreground" size={18} />
                  <select 
                    {...register("cardType")}
                    className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-border bg-background focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all appearance-none"
                  >
                    <option value="">Select...</option>
                    <option value="visa">Visa (Credit)</option>
                    <option value="mastercard">Mastercard (Credit)</option>
                    <option value="amex">American Express</option>
                    <option value="debit">Debit Card</option>
                    <option value="prepaid">Prepaid Card</option>
                  </select>
                </div>
                {errors.cardType && <p className="text-destructive text-xs">{errors.cardType.message}</p>}
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Cardholder Name</label>
              <div className="relative">
                <CreditCard className="absolute left-3 top-3 text-muted-foreground" size={18} />
                <input 
                  type="text" 
                  {...register("cardHolder")}
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-border bg-background focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all"
                  placeholder="Name on card"
                />
              </div>
              {errors.cardHolder && <p className="text-destructive text-xs">{errors.cardHolder.message}</p>}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Card Number</label>
              <div className="relative">
                <CreditCard className="absolute left-3 top-3 text-muted-foreground" size={18} />
                <input 
                  type="text" 
                  inputMode="numeric"
                  {...register("cardNumber")}
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl border border-border bg-background focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all font-mono"
                  placeholder="1234 5678 9012 3456"
                />
              </div>
              {errors.cardNumber && <p className="text-destructive text-xs">{errors.cardNumber.message}</p>}
            </div>

            <button 
              type="submit" 
              disabled={isPredicting}
              className="w-full bg-primary text-primary-foreground py-3 rounded-xl font-medium mt-4 flex items-center justify-center gap-2 hover:bg-primary/90 transition-all disabled:opacity-70 disabled:cursor-not-allowed"
            >
              {isPredicting ? (
                <>
                  <div className="w-5 h-5 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                  Running Model...
                </>
              ) : (
                'Evaluate Transaction'
              )}
            </button>
          </form>
        </motion.div>

        {/* Result Section */}
        <div className="relative">
          <AnimatePresence mode="wait">
            {!result && !isPredicting && (
              <motion.div 
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-full flex flex-col items-center justify-center p-12 text-center border-2 border-dashed border-border rounded-2xl bg-card/50"
              >
                <div className="w-16 h-16 bg-secondary rounded-full flex items-center justify-center mb-4">
                  <Bot size={32} className="text-muted-foreground" />
                </div>
                <h3 className="text-lg font-medium mb-2">Awaiting Data</h3>
                <p className="text-muted-foreground text-sm max-w-sm">
                  Enter transaction and card details and run the evaluation to see the model's prediction and quick card validity checks.
                </p>
              </motion.div>
            )}

            {isPredicting && (
              <motion.div 
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-full flex flex-col items-center justify-center p-12 text-center border border-border rounded-2xl bg-card shadow-sm"
              >
                <div className="relative w-24 h-24 mb-6">
                  <div className="absolute inset-0 border-4 border-secondary rounded-full"></div>
                  <div className="absolute inset-0 border-4 border-primary rounded-full border-t-transparent animate-spin"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <ShieldCheck size={28} className="text-primary animate-pulse" />
                  </div>
                </div>
                <h3 className="text-lg font-medium mb-2">Analyzing Patterns...</h3>
                <p className="text-muted-foreground text-sm">
                  Performing card validation and simulated ML scoring.
                </p>
              </motion.div>
            )}

            {result && !isPredicting && (
              <motion.div 
                key="result"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className={`h-full flex flex-col p-8 border rounded-2xl shadow-lg relative overflow-hidden ${
                  result.isFraud 
                    ? 'bg-destructive/5 border-destructive/20' 
                    : 'bg-safe/5 border-safe/20'
                }`}
              >
                {/* Decorative background blur */}
                <div className={`absolute -top-24 -right-24 w-48 h-48 rounded-full blur-3xl opacity-20 ${
                  result.isFraud ? 'bg-destructive' : 'bg-safe'
                }`} />

                <div className="flex items-center gap-3 mb-8">
                  <div className={`p-3 rounded-xl text-white ${
                    result.isFraud ? 'bg-destructive' : 'bg-safe'
                  }`}>
                    {result.isFraud ? <ShieldAlert size={28} /> : <ShieldCheck size={28} />}
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Evaluation Result</h3>
                    <p className={`text-2xl font-bold ${
                      result.isFraud ? 'text-destructive' : 'text-safe'
                    }`}>
                      {result.isFraud ? 'Fraudulent Transaction ❌' : 'Legitimate Transaction ✅'}
                    </p>
                  </div>
                </div>

                <div className="bg-card p-6 rounded-xl border border-border mb-6 shadow-sm">
                  <div className="flex justify-between items-end mb-2">
                    <span className="text-sm font-medium text-muted-foreground">Fraud Probability Score</span>
                    <span className="text-3xl font-mono font-bold">
                      {result.score.toFixed(3)}
                    </span>
                  </div>
                  <div className="w-full h-3 bg-secondary rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${result.score * 100}%` }}
                      transition={{ duration: 1, ease: "easeOut" }}
                      className={`h-full ${
                        result.isFraud ? 'bg-destructive' : 'bg-safe'
                      }`}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground mt-2 font-mono">
                    <span>0.00 (Safe)</span>
                    <span>1.00 (High Risk)</span>
                  </div>
                </div>

                <div className="flex-1">
                  <h4 className="text-sm font-medium mb-4 uppercase text-muted-foreground tracking-wider">Key Factors</h4>
                  <ul className="space-y-3">
                    {result.factors.map((factor, i) => (
                      <motion.li 
                        key={i}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 + 0.3 }}
                        className="flex items-start gap-2 text-sm"
                      >
                        <div className={`mt-1 w-1.5 h-1.5 rounded-full shrink-0 ${
                          result.isFraud ? 'bg-destructive' : 'bg-primary'
                        }`} />
                        {factor}
                      </motion.li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}