require("dotenv").config();
const express = require("express");
const apiRoutes = require("./src/routes/api");

const app = express();
const PORT = process.env.PORT || 3000;
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

app.use("/api", apiRoutes);

app.listen(PORT, () => {
    console.log(`S3 Upload Service running on port ${PORT}`);
});