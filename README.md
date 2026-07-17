# Unified Async LLM Client

Cliente asíncrono unificado para OpenAI y Anthropic con una interfaz común,
streaming de texto, configuración validada por Pydantic y manejo controlado de
errores transitorios.

## Inicio rápido

1. Cloná el repositorio y entrá al directorio:

   ```bash
   git clone https://github.com/thiagoooo34252/unified-async-llm-client.git
   cd unified-async-llm-client
   ```

2. Creá un entorno con Python 3.12 e instalá las dependencias:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

   También podés usar `uv`:

   ```bash
   uv sync --extra dev
   ```

3. Creá tu archivo local de configuración:

   ```bash
   cp .env.example .env
   ```

4. Agregá en `.env` la API key del proveedor elegido y ejecutá:

   ```bash
   python main.py
   ```

El script realiza una generación normal y otra en streaming con la pregunta
“¿Qué es la entropía?”. Ambas llamadas consumen créditos del proveedor.

## Configuración

| Variable | Descripción | Valor predeterminado |
|---|---|---|
| `LLM_PROVIDER` | Proveedor activo: `openai` o `anthropic` | `openai` |
| `OPENAI_API_KEY` | Credencial de OpenAI | Sin valor |
| `OPENAI_MODEL` | Modelo de OpenAI | `gpt-5.6-luna` |
| `ANTHROPIC_API_KEY` | Credencial de Anthropic | Sin valor |
| `ANTHROPIC_MODEL` | Modelo de Anthropic | `claude-sonnet-5` |
| `LLM_TEMPERATURE` | Aleatoriedad entre `0` y `2` | `0.2` |
| `LLM_MAX_TOKENS` | Máximo de tokens de salida | `1024` |
| `LLM_TIMEOUT_SECONDS` | Timeout por solicitud | `60` |
| `LLM_MAX_RETRIES` | Reintentos automáticos del SDK | `2` |

Para cambiar a Anthropic:

```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=tu_clave_local
```

## Arquitectura

| Archivo | Responsabilidad |
|---|---|
| `schemas.py` | Mensajes, configuración, respuestas y eventos validados |
| `clients/base.py` | Contrato común `BaseLLMClient` |
| `clients/openai_client.py` | Adaptador para OpenAI Responses API |
| `clients/anthropic_client.py` | Adaptador para Anthropic Messages API |
| `manager.py` | Selección dinámica del proveedor y API unificada |
| `main.py` | Carga del entorno y demostración normal/streaming |

`AsyncLLMManager` expone dos operaciones:

```python
response = await manager.generate(messages)

async for event in manager.stream(messages):
    if event.content:
        print(event.content, end="")
```

OpenAI y Anthropic tienen formatos distintos para instrucciones, mensajes,
respuestas y streaming. Cada adaptador traduce esos formatos al mismo contrato
de `ModelResponse`, `ErrorResponse` y `StreamEvent`.

La implementación de OpenAI usa Responses API porque es la API principal para
generación de texto en el SDK oficial actual. Conserva la misma asincronía y el
mismo streaming requeridos por la consigna.

## Errores controlados

Los SDKs reintentan automáticamente según `LLM_MAX_RETRIES`. Si el problema
persiste, el cliente devuelve errores tipados para:

- límites de tasa;
- timeouts;
- fallos de conexión;
- respuestas HTTP del proveedor;
- respuestas exitosas sin contenido textual.

Los mensajes controlados no incluyen API keys, cuerpos de respuesta ni detalles
internos sensibles. Los errores de programación inesperados no se silencian.

## Verificación

Las pruebas usan clientes simulados y no realizan llamadas reales:

```bash
pytest
ruff check .
ruff format --check .
pyright
```

GitHub Actions ejecuta estas mismas verificaciones automáticamente en cada push
y pull request, sin acceder a credenciales de proveedores.

### Validación contra APIs reales

La prueba de integración real está desactivada por defecto para evitar consumo
accidental. Requiere ambas claves en el `.env` local y realiza cuatro solicitudes
breves: generación normal y streaming para OpenAI y Anthropic.

```bash
RUN_LIVE_LLM_TESTS=1 pytest -m live -q
```

Las pruebas limitan cada respuesta a 64 tokens. Nunca deben agregarse las claves
al repositorio ni configurarse como secretos del workflow para esta entrega.

## Seguridad

- Nunca agregues una API key a `.env.example` ni al código.
- `.env` está excluido por `.gitignore`.
- Si una clave se publica accidentalmente, revocala y generá una nueva.
