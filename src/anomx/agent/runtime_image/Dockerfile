FROM node:22-bookworm-slim AS node
FROM python:3.12-slim-bookworm
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
COPY anomx/agent/runtime_image /opt/anomx-runtime-image
RUN sh /opt/anomx-runtime-image/install-tools.sh
COPY anomx /opt/anomx-source/anomx
ENV PYTHONPATH=/opt/anomx-source PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN mkdir -p /anomx/brain /anomx/skills /anomx/sessions \
    && python -c 'import anomx; import numpy; import pandas'
RUN printf '#!/bin/sh\nexec python -m anomx "$@"\n' > /usr/local/bin/anomx \
    && chmod +x /usr/local/bin/anomx
WORKDIR /anomx
CMD ["sleep", "infinity"]
