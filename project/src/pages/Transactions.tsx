import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Filter, MoreVertical, Download } from 'lucide-react';
import { MOCK_TRANSACTIONS } from '../lib/mockData';
import { formatCurrency } from '../lib/utils';

export default function Transactions() {
  const [searchTerm, setSearchTerm] = useState('');

  const filtered = MOCK_TRANSACTIONS.filter(tx => 
    tx.merchant.toLowerCase().includes(searchTerm.toLowerCase()) || 
    tx.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Transactions Directory</h1>
          <p className="text-muted-foreground mt-1">View and manage all processed card transactions.</p>
        </div>
        <div className="flex items-center gap-3 w-full sm:w-auto">
          <button className="flex items-center gap-2 px-4 py-2 bg-secondary text-secondary-foreground hover:bg-secondary/80 rounded-lg font-medium transition-colors">
            <Download size={18} />
            Export
          </button>
        </div>
      </div>

      <div className="bg-card rounded-2xl shadow-sm border border-border overflow-hidden flex flex-col">
        <div className="p-4 border-b border-border flex flex-col sm:flex-row gap-4 justify-between items-center bg-muted/20">
          <div className="relative w-full sm:max-w-md">
            <Search className="absolute left-3 top-2.5 text-muted-foreground" size={18} />
            <input 
              type="text" 
              placeholder="Search by ID or Merchant..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 rounded-lg border border-border bg-background focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all text-sm"
            />
          </div>
          <button className="flex items-center gap-2 px-3 py-2 border border-border rounded-lg text-sm font-medium hover:bg-secondary transition-colors w-full sm:w-auto justify-center">
            <Filter size={16} />
            More Filters
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-muted-foreground uppercase bg-secondary/50">
              <tr>
                <th className="px-6 py-4 font-medium">Transaction ID</th>
                <th className="px-6 py-4 font-medium">Date & Time</th>
                <th className="px-6 py-4 font-medium">Merchant</th>
                <th className="px-6 py-4 font-medium">Location</th>
                <th className="px-6 py-4 font-medium text-right">Amount</th>
                <th className="px-6 py-4 font-medium text-center">Score</th>
                <th className="px-6 py-4 font-medium">Status</th>
                <th className="px-6 py-4 font-medium"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {filtered.map((tx, i) => (
                <motion.tr 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                  key={tx.id} 
                  className="hover:bg-muted/50 transition-colors group"
                >
                  <td className="px-6 py-4 font-mono text-xs text-muted-foreground">{tx.id}</td>
                  <td className="px-6 py-4 whitespace-nowrap">{tx.time}</td>
                  <td className="px-6 py-4 font-medium">{tx.merchant}</td>
                  <td className="px-6 py-4 text-muted-foreground">{tx.location}</td>
                  <td className="px-6 py-4 text-right font-medium">{formatCurrency(tx.amount)}</td>
                  <td className="px-6 py-4 text-center">
                    <span className={`font-mono text-xs px-2 py-1 rounded bg-secondary ${
                      tx.score > 0.65 ? 'text-destructive font-bold' : 'text-muted-foreground'
                    }`}>
                      {tx.score.toFixed(3)}
                    </span>
                  </td>
                  <td className="px-6 py-4">
                    <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium border ${
                      tx.isFraud 
                        ? 'bg-destructive/10 text-destructive border-destructive/20' 
                        : 'bg-safe/10 text-safe border-safe/20'
                    }`}>
                      {tx.isFraud ? 'Fraudulent' : 'Legitimate'}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <button className="text-muted-foreground hover:text-foreground p-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <MoreVertical size={16} />
                    </button>
                  </td>
                </motion.tr>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={8} className="px-6 py-12 text-center text-muted-foreground">
                    No transactions found matching your search.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        
        <div className="p-4 border-t border-border flex items-center justify-between text-sm text-muted-foreground bg-muted/10">
          <span>Showing {filtered.length} of {MOCK_TRANSACTIONS.length} results</span>
          <div className="flex gap-2">
            <button className="px-3 py-1 border border-border rounded hover:bg-secondary disabled:opacity-50" disabled>Previous</button>
            <button className="px-3 py-1 border border-border rounded hover:bg-secondary">Next</button>
          </div>
        </div>
      </div>
    </div>
  );
}