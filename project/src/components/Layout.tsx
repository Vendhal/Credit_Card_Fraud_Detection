import React, { useState, useEffect } from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { 
  ShieldCheck, 
  LayoutDashboard, 
  Activity, 
  ListOrdered, 
  LogOut,
  Menu,
  X,
  Bell
} from 'lucide-react';
import { cn } from '../lib/utils';
import { motion, AnimatePresence } from 'framer-motion';
import { isAuthenticated, logout } from '../lib/auth';

const NAV_ITEMS = [
  { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/predict', label: 'Fraud Detection', icon: Activity },
  { path: '/transactions', label: 'Transactions', icon: ListOrdered },
];

export default function Layout() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    // If user is not authenticated, redirect to auth page
    if (!isAuthenticated()) {
      navigate('/auth', { replace: true });
    }
  }, [navigate]);

  const handleLogout = () => {
    logout();
    navigate('/auth', { replace: true });
  };

  const SidebarContent = () => (
    <>
      <div className="flex items-center gap-3 px-6 py-8">
        <div className="bg-primary text-primary-foreground p-2 rounded-xl">
          <ShieldCheck size={28} />
        </div>
        <div>
          <h1 className="font-bold text-lg leading-tight">Aegis</h1>
          <p className="text-xs text-muted-foreground font-medium">Fraud Detection</p>
        </div>
      </div>

      <nav className="flex-1 px-4 space-y-2">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            onClick={() => setIsMobileMenuOpen(false)}
            className={({ isActive }) => cn(
              "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 font-medium",
              isActive 
                ? "bg-primary text-primary-foreground shadow-md" 
                : "text-muted-foreground hover:bg-secondary hover:text-foreground"
            )}
          >
            <item.icon size={20} />
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-border mt-auto">
        <button 
          onClick={handleLogout}
          className="flex w-full items-center gap-3 px-4 py-3 rounded-xl text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors font-medium"
        >
          <LogOut size={20} />
          Sign Out
        </button>
      </div>
    </>
  );

  return (
    <div className="min-h-screen bg-background flex flex-col md:flex-row">
      {/* Mobile Header */}
      <header className="md:hidden flex items-center justify-between bg-card p-4 border-b border-border sticky top-0 z-20">
        <div className="flex items-center gap-2">
          <ShieldCheck className="text-primary" size={24} />
          <span className="font-bold">Aegis</span>
        </div>
        <button onClick={() => setIsMobileMenuOpen(true)} className="p-2">
          <Menu size={24} />
        </button>
      </header>

      {/* Mobile Sidebar Overlay */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <>
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsMobileMenuOpen(false)}
              className="fixed inset-0 bg-black/50 z-40 md:hidden"
            />
            <motion.aside 
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed inset-y-0 left-0 w-72 bg-card border-r border-border z-50 flex flex-col md:hidden shadow-2xl"
            >
              <div className="absolute right-4 top-6">
                <button onClick={() => setIsMobileMenuOpen(false)} className="p-2 text-muted-foreground">
                  <X size={20} />
                </button>
              </div>
              <SidebarContent />
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Desktop Sidebar */}
      <aside className="hidden md:flex w-72 bg-card border-r border-border flex-col sticky top-0 h-screen">
        <SidebarContent />
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Desktop Top Header */}
        <header className="hidden md:flex items-center justify-between px-8 py-5 bg-background/80 backdrop-blur-md sticky top-0 z-10 border-b border-transparent transition-all">
          <h2 className="text-xl font-semibold text-foreground capitalize">
            {window.location.pathname.replace('/', '') || 'Dashboard'}
          </h2>
          <div className="flex items-center gap-4">
            <button className="p-2 relative text-muted-foreground hover:text-foreground transition-colors">
              <Bell size={20} />
              <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-destructive rounded-full border border-card"></span>
            </button>
            <div className="flex items-center gap-3 pl-4 border-l border-border">
              <div className="w-9 h-9 rounded-full bg-primary/10 flex items-center justify-center text-primary font-bold">
                AD
              </div>
              <div className="text-sm">
                <p className="font-semibold leading-none">Admin User</p>
                <p className="text-muted-foreground text-xs mt-1">admin@aegis.io</p>
              </div>
            </div>
          </div>
        </header>

        <div className="flex-1 p-4 md:p-8 overflow-y-auto">
          <Outlet />
        </div>
      </main>
    </div>
  );
}