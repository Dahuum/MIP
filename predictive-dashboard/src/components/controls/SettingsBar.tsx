'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useI18n, Language } from '@/lib/i18n';
import { useTheme, Theme } from '@/lib/theme';

interface DropdownProps {
  trigger: React.ReactNode;
  children: React.ReactNode;
  align?: 'left' | 'right';
}

function Dropdown({ trigger, children, align = 'right' }: DropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 transition-all text-sm"
      >
        {trigger}
        <svg
          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className={`absolute top-full mt-2 ${align === 'right' ? 'right-0' : 'left-0'} 
              min-w-[140px] py-2 rounded-lg bg-slate-800 border border-white/10 shadow-xl z-50`}
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

interface DropdownItemProps {
  onClick: () => void;
  active?: boolean;
  children: React.ReactNode;
}

function DropdownItem({ onClick, active, children }: DropdownItemProps) {
  return (
    <button
      onClick={onClick}
      className={`w-full px-4 py-2 text-left text-sm flex items-center gap-2 transition-colors
        ${active ? 'bg-white/10 text-white' : 'text-gray-300 hover:bg-white/5 hover:text-white'}`}
    >
      {active && (
        <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      )}
      <span className={active ? '' : 'ml-6'}>{children}</span>
    </button>
  );
}

export function LanguageSelector() {
  const { language, setLanguage, t } = useI18n();

  const languages: { code: Language; label: string; flag: string }[] = [
    { code: 'en', label: 'English', flag: '🇬🇧' },
    { code: 'fr', label: 'Français', flag: '🇫🇷' },
  ];

  const current = languages.find(l => l.code === language)!;

  return (
    <Dropdown
      trigger={
        <>
          <span className="text-lg">{current.flag}</span>
          <span className="hidden sm:inline text-gray-200">{current.label}</span>
        </>
      }
    >
      {languages.map((lang) => (
        <DropdownItem
          key={lang.code}
          onClick={() => setLanguage(lang.code)}
          active={language === lang.code}
        >
          <span className="text-lg mr-2">{lang.flag}</span>
          {lang.label}
        </DropdownItem>
      ))}
    </Dropdown>
  );
}

export function ThemeSelector() {
  const { theme, setTheme } = useTheme();
  const { t } = useI18n();

  const themes: { code: Theme; label: string; icon: string }[] = [
    { code: 'dark', label: t.header.dark, icon: '🌙' },
    { code: 'light', label: t.header.light, icon: '☀️' },
  ];

  const current = themes.find(th => th.code === theme)!;

  return (
    <Dropdown
      trigger={
        <>
          <span className="text-lg">{current.icon}</span>
          <span className="hidden sm:inline text-gray-200">{current.label}</span>
        </>
      }
    >
      {themes.map((th) => (
        <DropdownItem
          key={th.code}
          onClick={() => setTheme(th.code)}
          active={theme === th.code}
        >
          <span className="text-lg mr-2">{th.icon}</span>
          {th.label}
        </DropdownItem>
      ))}
    </Dropdown>
  );
}

export function SettingsBar() {
  return (
    <div className="flex items-center gap-2">
      <LanguageSelector />
      <ThemeSelector />
    </div>
  );
}
