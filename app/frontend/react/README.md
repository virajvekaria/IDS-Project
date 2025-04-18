# Document Intelligence Search System (DISS) Frontend

This is the React frontend for the Document Intelligence Search System (DISS).

## Development

To start the development server:

```bash
cd app/frontend/react
npm install
npm start
```

This will start the development server at http://localhost:8080 with hot reloading.

## Building

To build the frontend for production:

```bash
cd app/frontend/react
npm install
npm run build
```

This will create a `dist` directory with the compiled assets.

## Project Structure

- `src/` - Source code
  - `components/` - React components
  - `services/` - API services
  - `styles/` - CSS styles
- `public/` - Static assets
- `dist/` - Compiled assets (created after build)

## Features

- Chat interface with streaming responses
- Document management
- Conversation history
- Responsive design
