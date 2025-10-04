import { MongoClient, ServerApiVersion } from 'mongodb';
import { config } from 'dotenv';

config();

const uri = process.env.MONGODB_URI;

// Create a MongoClient with a MongoClientOptions object
const client = new MongoClient(uri, {
    serverApi: {
        version: ServerApiVersion.v1,
        strict: true,
        deprecationErrors: true,
    }
});

let database = null;

export async function connectToDatabase() {
    try {
        if (!database) {
            await client.connect();
            database = client.db('ai_image_generator');
            console.log('✅ Successfully connected to MongoDB!');
            
            // Create indexes for better performance
            await database.collection('generations').createIndex({ "timestamp": -1 });
            await database.collection('generations').createIndex({ "prompt": "text" });
        }
        return database;
    } catch (error) {
        console.error('❌ MongoDB connection error:', error);
        throw error;
    }
}

export async function closeDatabaseConnection() {
    try {
        await client.close();
        console.log('📦 MongoDB connection closed');
    } catch (error) {
        console.error('Error closing database connection:', error);
    }
}

export function getDatabase() {
    if (!database) {
        throw new Error('Database not initialized. Call connectToDatabase first.');
    }
    return database;
}