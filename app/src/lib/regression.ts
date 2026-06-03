import { standardDeviation } from 'simple-statistics';

export interface Metadata {
  i: string[];
  p: string[];
  q: string[];
  c: string[];
  g: string[];
  t: string[];
}

export type PredictionsData = Record<string, {
  p: number;    // Predicted cutoff
  ci: [number, number]; // 90% Confidence Interval [low, high]
  mu: number;   // ML Bias
  sigma: number; // ML Uncertainty
}>;

// Statistical error function for normal distribution
function erf(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

function calculateProbability(rank: number, mean: number, sigma: number): number {
  const z = (mean - rank) / (sigma * Math.sqrt(2));
  const prob = 0.5 * (1 + erf(z));
  return Math.round(prob * 100);
}

export interface RankRecord {
  i: number; // institute ID
  p: number; // program ID
  q: number; // quota ID
  c: number; // category ID
  g: number; // gender ID
  cr: number; // closing rank
  y: number; // year
  r: number; // round
  t: number; // type ID
}

export interface RoundPrediction {
  round: number;
  predictedClose: number;
  probability: number;
  ci?: [number, number]; // Confidence interval
}

export interface PredictionResult {
  institute: string;
  program: string;
  type: string;
  quota: string;
  category: string;
  gender: string;
  mostLikelyRound: number;
  earliestRound: number;
  finalProbability: number;
  classification: "Safe" | "Likely" | "Competitive" | "Dream" | "Unlikely";
  roundChances: RoundPrediction[];
  history: {year: number, round: number, close: number}[];
  trend: "up" | "down" | "flat";
  rankMargin: number;
  predictedFinalClose: number;
  volatility: number;      // Dynamic % deviation
  reliability: number;     // 0-1 score based on data quality/R^2
}

function calculateWeightedLinearRegression(data: [number, number][]) {
  if (data.length < 2) return null;
  const maxYear = Math.max(...data.map(d => d[0]));
  const weightedPoints = data.map(d => {
    const weight = Math.pow(0.8, maxYear - d[0]);
    return { x: d[0], y: d[1], w: weight };
  });

  let sumW = 0, sumWX = 0, sumWY = 0, sumWXX = 0, sumWXY = 0;
  for (const p of weightedPoints) {
    sumW += p.w;
    sumWX += p.w * p.x;
    sumWY += p.w * p.y;
    sumWXX += p.w * p.x * p.x;
    sumWXY += p.w * p.x * p.y;
  }

  const denominator = (sumW * sumWXX - sumWX * sumWX);
  if (Math.abs(denominator) < 1e-10) return null;
  const m = (sumW * sumWXY - sumWX * sumWY) / denominator;
  const b = (sumWY - m * sumWX) / sumW;

  // Calculate R² (Coefficient of Determination) for reliability
  const yMean = sumWY / sumW;
  let ssRes = 0;
  let ssTot = 0;
  for (const p of weightedPoints) {
    const yPred = m * p.x + b;
    ssRes += p.w * Math.pow(p.y - yPred, 2);
    ssTot += p.w * Math.pow(p.y - yMean, 2);
  }
  const r2 = ssTot === 0 ? 1 : 1 - (ssRes / ssTot);

  return { m, b, r2 };
}

export function predictColleges(
  userRank: number,
  category: string,
  gender: string,
  records: RankRecord[],
  metadata: Metadata,
  predictions: PredictionsData | null = null
): PredictionResult[] {
  const maxYear = 2025; // Records go up to 2025
  const targetYear = maxYear + 1;

  // Convert input strings to IDs
  const categoryId = metadata.c.indexOf(category);
  const genderId = metadata.g.indexOf(gender);
  const genderNeutralId = metadata.g.indexOf('Gender-Neutral');

  if (categoryId === -1) return [];

  const filtered = records.filter(r => 
    r.c === categoryId && 
    (r.g === genderId || (gender === 'Gender-Neutral' && r.g === genderNeutralId))
  );
  
  const grouped = new Map<string, RankRecord[]>();
  for (const r of filtered) {
    const key = `${r.i}|${r.p}|${r.q}`;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key)!.push(r);
  }

  const results: PredictionResult[] = [];

  for (const [key, historyRecs] of grouped.entries()) {
    const split = key.split("|");
    const instId = parseInt(split[0]);
    const progId = parseInt(split[1]);
    const quotaId = parseInt(split[2]);
    
    const inst = metadata.i[instId];
    const prog = metadata.p[progId];
    const quota = metadata.q[quotaId];
    const type = metadata.t[historyRecs[0].t];

    const rounds = [1, 2, 3, 4, 5];
    const roundChances: RoundPrediction[] = [];
    
    // Deduplicate history to show only final round per year for smoother trajectory
    const historyMap = new Map<number, typeof historyRecs[0]>();
    for (const h of historyRecs) {
      const year = Number(h.y);
      const round = Number(h.r);
      if (!historyMap.has(year) || round > historyMap.get(year)!.r) {
        historyMap.set(year, { ...h, y: year, r: round });
      }
    }
    const historyData = Array.from(historyMap.values())
      .map(h => ({
        year: h.y,
        round: h.r,
        close: h.cr
      }))
      .sort((a, b) => a.year - b.year);

    for (const round of rounds) {
      const predKey = `${inst}|${prog}|${quota}|${category}|${gender}|${round}`;
      const mlPred = predictions ? predictions[predKey] : null;

      if (mlPred) {
        const mean = mlPred.p + mlPred.mu;
        const prob = calculateProbability(userRank, mean, mlPred.sigma);

        roundChances.push({
          round,
          predictedClose: mlPred.p,
          probability: Math.max(0, Math.min(100, prob)),
          ci: mlPred.ci
        });
        continue;
      }

      // Fallback to WLR if no ML prediction available
      const roundHistory = historyRecs
        .filter(r => r.r === round)
        .map(r => [r.y, r.cr] as [number, number])
        .sort((a, b) => a[0] - b[0]);

      if (roundHistory.length === 0) {
        if (roundChances.length > 0) {
          const last = roundChances[roundChances.length - 1];
          roundChances.push({
            round,
            predictedClose: last.predictedClose,
            probability: last.probability,
            ci: last.ci
          });
        }
        continue;
      }

      let predictedClose: number;
      let stdDev: number;
      const regression = calculateWeightedLinearRegression(roundHistory);
      if (regression) {
        predictedClose = Math.max(1, Math.round(regression.m * targetYear + regression.b));
        const residuals = roundHistory.map(h => h[1] - (regression.m * h[0] + regression.b));
        stdDev = standardDeviation(residuals) || (predictedClose * 0.05);
      } else {
        predictedClose = roundHistory[roundHistory.length - 1][1];
        stdDev = predictedClose * 0.08;
      }

      if (roundChances.length > 0 && predictedClose < roundChances[roundChances.length - 1].predictedClose) {
        predictedClose = roundChances[roundChances.length - 1].predictedClose;
      }

      const z = (predictedClose - userRank) / (stdDev + 1);
      let probability = Math.round(Math.min(99, Math.max(1, 100 / (1 + Math.exp(-1.7 * z)))));
      
      if (roundChances.length > 0 && probability < roundChances[roundChances.length - 1].probability) {
        probability = roundChances[roundChances.length - 1].probability;
      }

      const ciLow = Math.max(1, Math.round(predictedClose - 1.645 * stdDev));
      const ciHigh = Math.round(predictedClose + 1.645 * stdDev);

      roundChances.push({ 
        round, 
        predictedClose, 
        probability, 
        ci: [ciLow, ciHigh] 
      });
    }

    if (roundChances.length === 0) continue;

    const finalRound = roundChances[roundChances.length - 1];
    const earliestRound = roundChances.find(r => r.probability >= 50)?.round || 0;
    const mostLikelyRound = roundChances.find(r => r.probability >= 80)?.round || earliestRound;
    
    let classification: PredictionResult["classification"] = "Unlikely";
    if (finalRound.probability >= 85) classification = "Safe";
    else if (finalRound.probability >= 60) classification = "Likely";
    else if (finalRound.probability >= 35) classification = "Competitive";
    else if (finalRound.probability >= 15) classification = "Dream";

    // Final forensics calculations
    const finalHistoryRecs = historyRecs.filter(r => r.r === roundChances[roundChances.length - 1].round);
    const crValues = finalHistoryRecs.map(h => h.cr);
    const meanCr = crValues.reduce((a, b) => a + b, 0) / crValues.length;
    const stdCr = finalHistoryRecs.length > 1 ? standardDeviation(crValues) : meanCr * 0.1;
    const volatility = Number(((stdCr / meanCr) * 100).toFixed(1));
    
    // Reliability score based on R2 and number of years
    const regFinal = calculateWeightedLinearRegression(finalHistoryRecs.map(r => [r.y, r.cr]));
    let reliability = 0.5;
    if (regFinal) {
      const dataWeight = Math.min(1, finalHistoryRecs.length / 5);
      const r2Weight = Math.max(0.5, regFinal.r2);
      reliability = Number((dataWeight * r2Weight).toFixed(2));
    }

    // Trend calculation
    let trend: "up" | "down" | "flat" = "flat";
    if (regFinal) {
      trend = regFinal.m < -50 ? "up" : regFinal.m > 50 ? "down" : "flat";
    }

    results.push({
      institute: inst,
      program: prog,
      type,
      quota,
      category,
      gender,
      mostLikelyRound,
      earliestRound,
      finalProbability: finalRound.probability,
      classification,
      roundChances,
      history: historyData,
      trend,
      rankMargin: finalRound.predictedClose - userRank,
      predictedFinalClose: finalRound.predictedClose,
      volatility,
      reliability
    });
  }

  const classificationPriority: Record<string, number> = {
    "Competitive": 0,
    "Likely": 1,
    "Safe": 2,
    "Dream": 3,
    "Unlikely": 4
  };

  results.sort((a, b) => {
    const priorityA = classificationPriority[a.classification];
    const priorityB = classificationPriority[b.classification];
    if (priorityA !== priorityB) return priorityA - priorityB;
    return b.finalProbability - a.finalProbability || a.predictedFinalClose - b.predictedFinalClose;
  });
  return results;
}

/**
 * Runs a high-iteration client-side Monte Carlo simulation for ALL rounds
 * Also returns histogram data for the final round to visualize the distribution.
 */
export async function runDeepSimulation(
  userRank: number, 
  roundPredictions: { round: number, predictedClose: number }[], 
  volatilityPercentage: number,
  iterations: number = 50000
): Promise<{ 
  probs: Record<number, number>, 
  distribution: { x: number, y: number }[] 
}> {
  const probs: Record<number, number> = {};
  const finalPred = roundPredictions[roundPredictions.length - 1];
  const stdDev = (finalPred.predictedClose * (volatilityPercentage / 100));
  
  // Histogram setup for final round
  const bins = 40;
  const range = stdDev * 4; // 4 sigma range
  const minX = finalPred.predictedClose - range;
  const maxX = finalPred.predictedClose + range;
  const binWidth = (maxX - minX) / bins;
  const histogram = new Array(bins).fill(0).map((_, i) => ({
    x: Math.round(minX + i * binWidth),
    y: 0
  }));

  for (const pred of roundPredictions) {
    let successes = 0;
    const currentStdDev = (pred.predictedClose * (volatilityPercentage / 100));
    const isFinal = pred.round === finalPred.round;
    
    for (let i = 0; i < iterations; i++) {
      // Box-Muller transform for normal distribution
      const u1 = Math.random();
      const u2 = Math.random();
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      
      const simulatedClose = pred.predictedClose + z0 * currentStdDev;
      if (simulatedClose >= userRank) successes++;

      if (isFinal) {
        const binIdx = Math.floor((simulatedClose - minX) / binWidth);
        if (binIdx >= 0 && binIdx < bins) {
          histogram[binIdx].y++;
        }
      }
    }
    probs[pred.round] = Math.round((successes / iterations) * 100);
  }
  
  // Normalize histogram for easier visualization
  const maxFreq = Math.max(...histogram.map(h => h.y));
  const distribution = histogram.map(h => ({
    x: h.x,
    y: Number(((h.y / maxFreq) * 100).toFixed(1))
  }));
  
  return { probs, distribution };
}

