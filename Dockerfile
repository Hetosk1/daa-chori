FROM node:latest

WORKDIR /app
COPY package*.json ./
RUN npm install --production
COPY . .
EXPOSE 46750
CMD ["node", "index.js"]

