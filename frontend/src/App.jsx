import { useState } from "react";
import "./styles.css";

const heroStats = [
  { value: "125", label: "vacas monitoreadas" },
  { value: "14", label: "variables analizadas" },
  { value: "semana 32", label: "última actualización" },
];

const filterOptions = [
  { id: "production", label: "Producción diaria" },
  { id: "quality", label: "Calidad de leche" },
  { id: "body", label: "Condición corporal" },
  { id: "feed", label: "Eficiencia alimenticia" },
  { id: "health", label: "Índice de salud" },
];

const indicators = [
  "Días en leche",
  "Calidad de proteína",
  "Índice de fertilidad",
  "Consumo de alimento",
];

const highlightCards = [
  {
    title: "Top producción de leche",
    entries: [
      { id: "V-203", metric: "43.8 L/día" },
      { id: "V-155", metric: "41.2 L/día" },
      { id: "V-089", metric: "39.7 L/día" },
    ],
  },
  {
    title: "Mejor índice de fertilidad",
    entries: [
      { id: "V-114", metric: "92 %" },
      { id: "V-222", metric: "89 %" },
      { id: "V-078", metric: "87 %" },
    ],
  },
  {
    title: "Mayor contenido de proteína",
    entries: [
      { id: "V-301", metric: "3.8 %" },
      { id: "V-178", metric: "3.6 %" },
      { id: "V-044", metric: "3.5 %" },
    ],
  },
];

const rankingRows = [
  {
    position: 1,
    id: "V-203",
    name: "Sue",
    milkingsPerDay: "3.2",
    averageLiters: "43.8",
    bodyCondition: "3.75",
    healthIndex: 97,
  },
  {
    position: 2,
    id: "V-155",
    name: "Bela",
    milkingsPerDay: "3.0",
    averageLiters: "41.2",
    bodyCondition: "3.60",
    healthIndex: 95,
  },
  {
    position: 3,
    id: "V-089",
    name: "Mora",
    milkingsPerDay: "2.9",
    averageLiters: "39.7",
    bodyCondition: "3.55",
    healthIndex: 94,
  },
  {
    position: 4,
    id: "V-114",
    name: "Rita",
    milkingsPerDay: "3.1",
    averageLiters: "38.5",
    bodyCondition: "3.50",
    healthIndex: 93,
  },
  {
    position: 5,
    id: "V-301",
    name: "Luna",
    milkingsPerDay: "2.8",
    averageLiters: "37.9",
    bodyCondition: "3.65",
    healthIndex: 91,
  },
  {
    position: 6,
    id: "V-078",
    name: "Roma",
    milkingsPerDay: "2.9",
    averageLiters: "37.0",
    bodyCondition: "3.40",
    healthIndex: 90,
  },
  {
    position: 7,
    id: "V-044",
    name: "Dora",
    milkingsPerDay: "2.7",
    averageLiters: "36.5",
    bodyCondition: "3.55",
    healthIndex: 88,
  },
  {
    position: 8,
    id: "V-178",
    name: "Pola",
    milkingsPerDay: "2.8",
    averageLiters: "35.9",
    bodyCondition: "3.52",
    healthIndex: 87,
  },
  {
    position: 9,
    id: "V-222",
    name: "Nora",
    milkingsPerDay: "2.6",
    averageLiters: "35.1",
    bodyCondition: "3.48",
    healthIndex: 86,
  },
  {
    position: 10,
    id: "V-310",
    name: "Tina",
    milkingsPerDay: "2.7",
    averageLiters: "34.7",
    bodyCondition: "3.42",
    healthIndex: 84,
  },
];

const insights = [
  {
    tag: "Alerta temprana",
    title: "Variaciones en conductividad",
    description:
      "Cuatro vacas del top bajaron más de 6 % en conductividad eléctrica en la última semana, posible indicador de mastitis.",
  },
  {
    tag: "Oportunidad",
    title: "Promover genética elite",
    description:
      "Las vacas V-203 y V-155 concentran la mejor combinación de leche + proteína, candidatas ideales para transferencia de embriones.",
  },
  {
    tag: "Plan de acción",
    title: "Optimizar raciones",
    description:
      "Integrar más forraje de alta fibra para las vacas con condición corporal < 3.4 sin perder volumen de ordeño.",
  },
];

function App() {
  const [activeFilter, setActiveFilter] = useState(filterOptions[0].id);

  return (
    <>
      <header className="hero">
        <div className="hero__content">
          <p className="hero__tag">Panel Operativo</p>
          <h1>Top de vacas según productividad, salud y genética</h1>
          <p className="hero__desc">
            Explora el desempeño histórico de cada vaca y detecta aquellas con mayor potencial para los programas de
            reproducción y ordeño.
          </p>
          <div className="hero__stats">
            {heroStats.map((stat) => (
              <div key={stat.label}>
                <p className="hero__stats-value">{stat.value}</p>
                <p className="hero__stats-label">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
        <div className="hero__glass">
          <p>Indicadores clave</p>
          <ul>
            {indicators.map((indicator) => (
              <li key={indicator}>{indicator}</li>
            ))}
          </ul>
        </div>
      </header>

      <main>
        <section className="filters">
          <div className="filters__bar">
            <p className="filters__label">Filtrar ranking por:</p>
            <div className="filters__chips">
              {filterOptions.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  className={`chip ${activeFilter === option.id ? "is-active" : ""}`}
                  onClick={() => setActiveFilter(option.id)}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="grid">
          {highlightCards.map((card) => (
            <article className="grid__card" key={card.title}>
              <h2>{card.title}</h2>
              <ol>
                {card.entries.map((entry) => (
                  <li key={entry.id}>
                    <span className="cow-id">{entry.id}</span>
                    {entry.metric}
                  </li>
                ))}
              </ol>
            </article>
          ))}
        </section>

        <section className="table-card">
          <header>
            <div>
              <p className="tag">Ranking general</p>
              <h2>Top 10 integral</h2>
            </div>
            <button className="ghost-btn" type="button">
              Exportar CSV
            </button>
          </header>
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Pos.</th>
                  <th>Vaca</th>
                  <th>Ordeños/día</th>
                  <th>Promedio (L)</th>
                  <th>Condición corporal</th>
                  <th>Índice salud</th>
                </tr>
              </thead>
              <tbody>
                {rankingRows.map((row) => (
                  <tr key={row.id}>
                    <td>{row.position}</td>
                    <td className="cow-cell">
                      <span className="avatar">{row.id}</span>
                      <strong>{row.name}</strong>
                    </td>
                    <td>{row.milkingsPerDay}</td>
                    <td>{row.averageLiters}</td>
                    <td>{row.bodyCondition}</td>
                    <td>{row.healthIndex}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="insights">
          {insights.map((item) => (
            <article key={item.title}>
              <p className="tag">{item.tag}</p>
              <h3>{item.title}</h3>
              <p>{item.description}</p>
            </article>
          ))}
        </section>
      </main>

      <footer>
        <p>Concentración IA · Última sincronización 05/08 · Datos simulados para prototipo visual.</p>
      </footer>
    </>
  );
}

export default App;
