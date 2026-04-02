"use client";

import { useState } from "react";

const BASE_URL = "/api";
const DEFAULT_SEED = 42;

const defaultAction = {
  price_adjustment: 0,
  marketing_spend: 100,
  restock_amount: 10,
  product_focus: "food"
};

const shopThemes = {
  cafe: {
    label: "Cafe",
    accent: "#9d4a27",
    facade: "linear-gradient(180deg, #f3c47d 0%, #d07b43 100%)"
  },
  food: {
    label: "Food Hall",
    accent: "#bc5c35",
    facade: "linear-gradient(180deg, #f5a768 0%, #d86139 100%)"
  },
  tech: {
    label: "Tech Kiosk",
    accent: "#2d7581",
    facade: "linear-gradient(180deg, #8ed1dd 0%, #3b8291 100%)"
  },
  stationary: {
    label: "Stationery",
    accent: "#5f5a99",
    facade: "linear-gradient(180deg, #bfbcf7 0%, #6e69b6 100%)"
  }
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function buildScene(result, action) {
  const observation = result?.observation ?? {
    day: 1,
    phase: "morning",
    shop_traffic: 18,
    conversion_rate: 0.22,
    revenue: 420,
    customer_satisfaction: 0.55,
    inventory_level: 0.68,
    monthly_budget: 10000,
    awareness: 0.4,
    market_sentiment: 0.5,
    competitor_pressure: 0.28,
    trend_factor: 1
  };
  const info = result?.info ?? {};
  const focus = action.product_focus;
  const focusTheme = shopThemes[focus] ?? shopThemes.food;
  const traffic = observation.shop_traffic;
  const visibleStudents = clamp(Math.ceil(traffic / 2.5), 6, 24);
  const entrantCount = Math.max(2, Math.round(visibleStudents * observation.conversion_rate));
  const outboundCount = Math.max(2, Math.round(visibleStudents * 0.28));
  const queueCount = Math.max(1, Math.round(visibleStudents * (0.15 + observation.competitor_pressure * 0.3)));
  const mainShopScale = 1 + (observation.awareness * 0.08);
  const competitorGlow = 0.25 + observation.competitor_pressure * 0.65;
  const trendLabel = typeof info.trend === "string" ? info.trend : "normal";

  const students = Array.from({ length: visibleStudents }, (_, index) => {
    const lane = index % 3;
    const phaseType = index < entrantCount ? "entering" : index < entrantCount + outboundCount ? "leaving" : "wandering";
    const hue = phaseType === "entering" ? 28 : phaseType === "leaving" ? 196 : 92;
    const xOffset = (index * 13) % 100;
    const delay = `${(index % 8) * 0.35}s`;
    const duration = `${4.8 + (index % 5) * 0.7}s`;
    const scale = 0.82 + (index % 4) * 0.08;
    const startX = phaseType === "entering" ? `${6 + xOffset * 0.22}%` : phaseType === "leaving" ? "48%" : `${12 + xOffset * 0.58}%`;
    const endX = phaseType === "entering" ? "48%" : phaseType === "leaving" ? `${72 + (xOffset % 18)}%` : `${18 + xOffset * 0.54}%`;
    const baseY = `${60 + lane * 10}%`;
    const driftY = `${phaseType === "wandering" ? 8 + (index % 4) * 3 : 0}%`;

    return {
      id: `student-${index}`,
      phaseType,
      style: {
        "--student-color": `hsl(${hue} 72% ${phaseType === "wandering" ? 50 : 58}%)`,
        "--student-shadow": `hsla(${hue} 80% 40% / 0.28)`,
        "--student-delay": delay,
        "--student-duration": duration,
        "--student-start-x": startX,
        "--student-end-x": endX,
        "--student-y": baseY,
        "--student-drift-y": driftY,
        "--student-scale": scale.toFixed(2)
      }
    };
  });

  const competitorShops = [
    {
      id: "left-competitor",
      name: "East Gate Bites",
      mood: observation.competitor_pressure > 0.55 ? "Aggressive promos" : "Moderate traffic",
      style: {
        "--shop-x": "16%",
        "--shop-y": "22%",
        "--shop-accent": "#d07a55",
        "--shop-glow": competitorGlow.toFixed(2)
      }
    },
    {
      id: "right-competitor",
      name: "Campus Tech Stop",
      mood: observation.competitor_pressure > 0.4 ? "Discount campaign" : "Light competition",
      style: {
        "--shop-x": "78%",
        "--shop-y": "18%",
        "--shop-accent": "#5d86af",
        "--shop-glow": (competitorGlow * 0.86).toFixed(2)
      }
    }
  ];

  return {
    observation,
    info,
    focusTheme,
    trendLabel,
    visibleStudents,
    entrantCount,
    queueCount,
    mainShopScale: mainShopScale.toFixed(2),
    competitorShops,
    students
  };
}

function MetricCard({ label, value, tone = "default" }) {
  return (
    <div className={`metric-card metric-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function HomePage() {
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("Ready to simulate the campus market.");
  const [loading, setLoading] = useState(false);
  const [seed, setSeed] = useState(DEFAULT_SEED);
  const [action, setAction] = useState(defaultAction);

  const scene = buildScene(result, action);

  async function resetEnv() {
    setLoading(true);
    setStatus("Rebuilding the market day...");
    try {
      const response = await fetch(`${BASE_URL}/reset`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed })
      });
      const data = await response.json();
      setResult(data);
      setStatus(`Episode reset with seed ${seed}.`);
    } catch (error) {
      setStatus(`Reset failed: ${String(error)}`);
    } finally {
      setLoading(false);
    }
  }

  async function stepEnv() {
    setLoading(true);
    setStatus("Students are reacting to the new shop strategy...");
    try {
      const response = await fetch(`${BASE_URL}/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(action)
      });
      const data = await response.json();
      setResult(data);
      setStatus("Step complete. Watch the new crowd pattern.");
    } catch (error) {
      setStatus(`Step failed: ${String(error)}`);
    } finally {
      setLoading(false);
    }
  }

  function updateField(name, value) {
    setAction((current) => ({
      ...current,
      [name]: value
    }));
  }

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Live Campus Simulation</p>
          <h1>Watch students flow through a 2D campus market.</h1>
          <p className="lede">
            The scene below turns each RL step into a small street simulation with shoppers
            entering, drifting, leaving, and reacting to competition, awareness, and shop focus.
          </p>
        </div>

        <div className="hero-toolbar">
          <label className="compact-field">
            Seed
            <input
              type="number"
              value={seed}
              onChange={(event) => setSeed(Number(event.target.value))}
            />
          </label>

          <div className="button-row">
            <button onClick={resetEnv} disabled={loading}>
              Reset Scene
            </button>
            <button onClick={stepEnv} disabled={loading}>
              Advance Step
            </button>
          </div>
        </div>

        <p className="status">{status}</p>
      </section>

      <section className="campus-stage panel">
        <div className="stage-head">
          <div>
            <p className="eyebrow">2D Ground View</p>
            <h2>Campus Street</h2>
          </div>
          <div className="stage-tags">
            <span>Day {scene.observation.day}</span>
            <span>{scene.observation.phase}</span>
            <span>{scene.trendLabel}</span>
          </div>
        </div>

        <div className="ground-scene">
          <div className="sky-glow" />
          <div className="street-lane" />
          <div className="crosswalk" />
          <div className="campus-garden campus-garden-left" />
          <div className="campus-garden campus-garden-right" />

          {scene.competitorShops.map((shop) => (
            <div key={shop.id} className="competitor-shop" style={shop.style}>
              <div className="shop-roof" />
              <div className="shop-body">
                <strong>{shop.name}</strong>
                <span>{shop.mood}</span>
              </div>
            </div>
          ))}

          <div
            className="main-shop"
            style={{
              "--shop-scale": scene.mainShopScale,
              "--shop-accent": scene.focusTheme.accent,
              "--shop-facade": scene.focusTheme.facade
            }}
          >
            <div className="main-shop-awning" />
            <div className="main-shop-body">
              <div className="shop-sign">{scene.focusTheme.label}</div>
              <div className="shop-window" />
              <div className="shop-door" />
            </div>
          </div>

          <div className="queue-badge">
            <span>Queue</span>
            <strong>{scene.queueCount}</strong>
          </div>

          {scene.students.map((student) => (
            <div
              key={student.id}
              className={`student student-${student.phaseType}`}
              style={student.style}
            />
          ))}
        </div>
      </section>

      <section className="metrics-grid">
        <MetricCard label="Traffic" value={scene.observation.shop_traffic} tone="warm" />
        <MetricCard
          label="Conversion"
          value={formatPercent(scene.observation.conversion_rate)}
          tone="cool"
        />
        <MetricCard
          label="Satisfaction"
          value={formatPercent(scene.observation.customer_satisfaction)}
          tone="good"
        />
        <MetricCard
          label="Competitor Pressure"
          value={formatPercent(scene.observation.competitor_pressure)}
          tone="risk"
        />
        <MetricCard
          label="Awareness"
          value={formatPercent(scene.observation.awareness)}
          tone="cool"
        />
        <MetricCard
          label="Revenue"
          value={`$${Math.round(scene.observation.revenue)}`}
          tone="good"
        />
      </section>

      <section className="panel-grid">
        <div className="panel action-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Control Inputs</p>
              <h2>Shop Strategy</h2>
            </div>
          </div>

          <label>
            Price Adjustment
            <input
              type="number"
              step="0.05"
              min="-1"
              max="1"
              value={action.price_adjustment}
              onChange={(event) => updateField("price_adjustment", Number(event.target.value))}
            />
          </label>
          <label>
            Marketing Spend
            <input
              type="number"
              step="10"
              min="0"
              value={action.marketing_spend}
              onChange={(event) => updateField("marketing_spend", Number(event.target.value))}
            />
          </label>
          <label>
            Restock Amount
            <input
              type="number"
              step="1"
              min="0"
              value={action.restock_amount}
              onChange={(event) => updateField("restock_amount", Number(event.target.value))}
            />
          </label>
          <label>
            Product Focus
            <select
              value={action.product_focus}
              onChange={(event) => updateField("product_focus", event.target.value)}
            >
              <option value="cafe">Cafe</option>
              <option value="food">Food</option>
              <option value="tech">Tech</option>
              <option value="stationary">Stationery</option>
            </select>
          </label>
        </div>

        <div className="panel insight-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">Simulation Readout</p>
              <h2>Market Pulse</h2>
            </div>
          </div>

          <div className="insight-list">
            <div className="insight-row">
              <span>Visible students</span>
              <strong>{scene.visibleStudents}</strong>
            </div>
            <div className="insight-row">
              <span>Entrants moving to your shop</span>
              <strong>{scene.entrantCount}</strong>
            </div>
            <div className="insight-row">
              <span>Inventory level</span>
              <strong>{formatPercent(scene.observation.inventory_level)}</strong>
            </div>
            <div className="insight-row">
              <span>Budget left</span>
              <strong>${Math.round(scene.observation.monthly_budget)}</strong>
            </div>
            <div className="insight-row">
              <span>Reward</span>
              <strong>{result ? result.reward.toFixed(2) : "0.00"}</strong>
            </div>
          </div>

          <pre>{result ? JSON.stringify(result, null, 2) : "No result yet."}</pre>
        </div>
      </section>
    </main>
  );
}
