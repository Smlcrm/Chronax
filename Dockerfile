# Build and serve the Chronax documentation on Cloud Run.
#
# The Cloud Build trigger builds this with the repo as the context, so `.` is the
# source we document. It pulls docgen (Smlcrm/web-doc-generator) at build time,
# generates the site from this repo, and serves it. docgen auto-loads the
# docgen.toml at the repo root for the project name and theme.
#
# Keyless by default (--hybrid: real cached pages where present, previews else).
# For full real docs, provide GEMINI_API_KEY and drop --hybrid.

FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Fetch docgen. While web-doc-generator is private, pass a token:
#   --build-arg DOCGEN_TOKEN=<github PAT>
ARG DOCGEN_REF=main
ARG DOCGEN_TOKEN=""
RUN if [ -n "$DOCGEN_TOKEN" ]; then \
        git clone --depth 1 --branch "$DOCGEN_REF" \
          "https://x-access-token:${DOCGEN_TOKEN}@github.com/Smlcrm/web-doc-generator.git" /docgen ; \
    else \
        git clone --depth 1 --branch "$DOCGEN_REF" \
          "https://github.com/Smlcrm/web-doc-generator.git" /docgen ; \
    fi
RUN pip install --no-cache-dir -r /docgen/requirements.txt
ENV PYTHONPATH=/docgen

# Generate the site from this repo (docgen.toml at the root is auto-loaded).
# With a key: full real docs. Without: keyless --hybrid fallback so the build
# never fails for lack of a key. The Cloud Build trigger supplies the key via a
# _GEMINI_API_KEY substitution passed as this build arg.
ARG GEMINI_API_KEY=""
COPY . /src
RUN if [ -n "$GEMINI_API_KEY" ]; then \
        echo "Generating real docs (Gemini)..." ; \
        GEMINI_API_KEY="$GEMINI_API_KEY" python -m docgen --repo /src --out /docgen/site ; \
    else \
        echo "No GEMINI_API_KEY set, building keyless --hybrid preview..." ; \
        python -m docgen --repo /src --out /docgen/site --hybrid ; \
    fi

ENV PORT=8080
EXPOSE 8080
CMD ["python", "/docgen/serve.py"]
