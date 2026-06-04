import { ModelTier } from '../types.js';

export enum BudgetWindow {
  PerSession = 'per_session',
  PerHour = 'per_hour',
  PerDay = 'per_day',
  PerMonth = 'per_month',
}

export interface BudgetLimit {
  window: BudgetWindow;
  tokens?: number;
  usd?: number;
}

export interface AlertConfig {
  slackWebhook?: string;
  logToStderr: boolean;
}

export class BudgetConfig {
  perSession?: BudgetLimit;
  perUser?: BudgetLimit;
  perAgent?: BudgetLimit;
  globalLimit?: BudgetLimit;
  alertAtPct: number = 0.8;
  alerts: AlertConfig = { logToStderr: true };
}
