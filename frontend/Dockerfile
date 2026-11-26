# Stage 1: build the React app
FROM node:22-alpine AS build
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: serve it with Nginx and forward /api to the backend
FROM nginx:1.27-alpine
COPY --from=build /web/dist /usr/share/nginx/html
# The official image fills ${API_KEY} in this template when the container starts.
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
EXPOSE 80
