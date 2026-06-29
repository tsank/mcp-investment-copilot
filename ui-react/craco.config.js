const path = require("path");

module.exports = {
  webpack: {
    alias: {
      "plotly.js/dist/plotly": path.resolve(
        __dirname,
        "node_modules/plotly.js-dist-min/plotly.min.js"
      ),
    },
  },
};
