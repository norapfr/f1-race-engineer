# F1 Race Engineer AI + Strategy Lab

Sistema de apoyo a decisiones para Fórmula 1: datos → features → modelos ML → simulador Monte Carlo →
optimizador de estrategia → AI Race Engineer. El LLM interpreta resultados de los modelos; nunca genera números.

> Proyecto personal sin relación con la FIA, la F1 ni sus empresas. Usa datos públicos no oficiales (FastF1, Jolpica-F1).

## Estado

| Fase | Contenido | Estado |
|---|---|---|
| 1 | Ingestión (Jolpica + FastF1), capas raw/clean en DuckDB, política anti-fuga | hecha |
| 2 | Features, baselines, modelos de predicción, evaluación temporal | hecha (primera versión) |
| 3 | Simulador Monte Carlo (distribución de posiciones) | pendiente |
| 4-8 | Degradación de neumáticos, Strategy Lab, AI Race Engineer, Race Replay, Live | pendiente |

## Uso (en este orden)

    pip install -e ".[dev,ingest]"
    python -m pytest
    python -m pipelines.ingest_results --from 2018 --to 2026            # resultados y clasificación (Jolpica)
    python -m pipelines.ingest_fastf1 --from 2018 --to 2026             # vueltas, meteo, dirección de carrera (FastF1)
    python -m pipelines.build_pace data/f1.duckdb                       # ritmo de carrera por piloto
    python -m pipelines.build_features data/f1.duckdb                   # tabla de features
    python -m pipelines.evaluate_baselines data/f1.duckdb
    python -m pipelines.train_eval data/f1.duckdb                       # modelos, ablación, calibración
    python -m pipelines.predict_race 2026_16                            # probabilidades de una carrera

La descarga de FastF1 es lenta y reanudable (límite de 500 llamadas/hora): si se corta, repite el mismo comando.

## Estructura

    ingestion/   fuentes → raw → clean (Jolpica, FastF1)      features/   features, ritmo, baselines, registro anti-fuga
    storage/     esquema DuckDB                                models/     modelos, evaluación, predicción de carrera
    pipelines/   comandos de extremo a extremo                 tests/      incluye tests de fuga por perturbación

## Anti-fuga de información

`features/availability.py` clasifica cada columna como PRE_WEEKEND, POST_QUALI o POST_RACE y falla si una feature no está
registrada o es posterior a la carrera. Los tests alteran los resultados de una carrera y comprueban que sus propias features y
predicciones no cambian.

## Resultados de la validación temporal (130 carreras de test, 2021-2026)

Entrenando siempre con temporadas anteriores. Referencia: la posición de clasificación.

| Modelo | Log loss victoria | Error medio de posición |
|---|---|---|
| Azar | 0,197 | 5,06 |
| Parrilla | 0,112 | 3,54 |
| Clasificación | 0,109 | 3,48 |
| Conjunto (logística + LightGBM) | 0,097 | 3,28 |

- La mejora en **posición final esperada** es robusta (0,1-0,2 puestos, intervalos de confianza por bootstrap fuera del cero, también sin 2023).
- La mejora en **probabilidad de victoria** se concentra en 2023, un año dominado por un solo coche; sin él no se distingue de la clasificación.
- Las probabilidades están comprimidas hacia el centro (conservadoras en los extremos); en podio la calibración es buena.
- Las features de ritmo de carrera ayudan algo a la regresión logística y no a LightGBM.