# Unified Async LLM Client

Cliente asíncrono unificado para trabajar con OpenAI y Anthropic mediante una única interfaz.

El proyecto implementa generación de texto normal y por streaming, selección de proveedor en tiempo de ejecución, validación con Pydantic y manejo controlado de errores.

## Características

- Interfaz abstracta común para múltiples proveedores.
- Implementaciones asíncronas con `AsyncOpenAI` y `AsyncAnthropic`.
- Soporte para respuestas completas y streaming incremental.
- Modelos Pydantic para validar solicitudes, respuestas y eventos.
- Errores normalizados para credenciales, límites de tasa, timeouts, conexión y respuestas vacías.
- Reintentos configurables delegados a los SDK oficiales.
- Pruebas unitarias sin llamadas reales a APIs externas.

## Requisitos

- Python 3.12 o superior.
- Una clave válida de OpenAI, Anthropic o ambas.
- [`uv`](https://docs.astral.sh/uv/) recomendado para instalar dependencias.

## Instalación

```bash
git clone https://github.com/thiagoooo34252/unified-async-llm-client.git
cd unified-async-llm-client
uv sync --extra dev
```

También puede instalarse con `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

En Windows, active el entorno con `.venv\Scripts\activate`.

## Configuración

Copiar el archivo de ejemplo y completar únicamente las claves disponibles:

```bash
cp .env.example .env
```

```dotenv
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_MODEL=gpt-5.6-luna
ANTHROPIC_MODEL=claude-sonnet-5
LLM_MAX_RETRIES=2
LLM_TIMEOUT_SECONDS=30
```

`.env` está excluido de Git. Nunca deben versionarse claves reales.

## Uso

Ejecutar la demostración:

```bash
uv run python main.py
```

El programa pregunta qué proveedor utilizar y envía la consulta:

```text
¿Qué es la entropía?
```

También muestra la respuesta mediante streaming en tiempo real.

### Uso programático

```python
import asyncio

from manager import AsyncLLMManager
from schemas import GenerationRequest, Provider


async def main() -> None:
    async with AsyncLLMManager() as manager:
        request = GenerationRequest(
            prompt="¿Qué es la entropía?",
            max_tokens=300,
        )

        response = await manager.generate(Provider.OPENAI, request)
        print(response)

        async for event in manager.stream(Provider.ANTHROPIC, request):
            if event.delta:
                print(event.delta, end="", flush=True)


asyncio.run(main())
```

Las operaciones devuelven modelos tipados:

- `GenerationResponse` cuando la generación finaliza correctamente.
- `ErrorResponse` cuando ocurre un error esperado.
- `StreamEvent` para cada fragmento, finalización o error de streaming.

## Arquitectura

```text
.
├── clients/
│   ├── base.py
│   ├── openai_client.py
│   └── anthropic_client.py
├── tests/
│   ├── test_clients.py
│   ├── test_manager.py
│   └── test_schemas.py
├── main.py
├── manager.py
└── schemas.py
```

`BaseLLMClient` define el contrato común. Cada cliente traduce las respuestas y excepciones del SDK a los modelos del dominio. `AsyncLLMManager` administra las instancias y permite seleccionar el proveedor sin cambiar el código consumidor.

## Calidad

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

## Seguridad

- Las claves se cargan exclusivamente desde variables de entorno.
- `.env` no se incluye en el repositorio.
- Los mensajes de error no exponen claves ni valores sensibles.
- Las solicitudes de prueba usan clientes simulados y no consumen créditos.

## Licencia

MIT. Ver [LICENSE](LICENSE).
