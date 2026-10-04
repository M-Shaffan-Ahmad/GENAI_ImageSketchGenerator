export default {
  server: {proxy: {'/api': {target: 'http://127.0.0.1:8000', rewrite: path => path.replace(/^\/api/, '')}}},
  preview: {proxy: {'/api': {target: 'http://127.0.0.1:8000', rewrite: path => path.replace(/^\/api/, '')}}}
};
