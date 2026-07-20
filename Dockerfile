# Serve-only image for Cloud Run. The docs are generated in a Cloud Build step
# (see cloudbuild.yaml) which writes ./site and ./serve.py into the build
# context, so this image just serves the prebuilt static site.
#
# Generation is kept out of the image on purpose: the Gemini key stays in the
# build step (never an image layer), and the persistent GCS doc cache lives in
# the build step too, so only changed modules are regenerated.
FROM python:3.12-slim
WORKDIR /app
COPY site /app/site
COPY serve.py /app/serve.py
ENV PORT=8080
EXPOSE 8080
CMD ["python", "/app/serve.py"]
