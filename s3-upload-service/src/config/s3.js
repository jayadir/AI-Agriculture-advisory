const {S3Client:Client}=require('@aws-sdk/client-s3');
const dotenv=require('dotenv');
dotenv.config();

const S3Client=new Client({
    region:process.env.AWS_REGION,
    credentials:{
        accessKeyId:process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey:process.env.AWS_SECRET_ACCESS_KEY
    }
});
module.exports=S3Client;