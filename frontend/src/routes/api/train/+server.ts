import type { RequestHandler } from './$types';

export const POST: RequestHandler = (async ({request}) => {
    const inputs = await request.json();

    console.log("Starting training process (/api/train/server.ts route)...");
    console.log("Inputs received for training:", inputs);

    // const response = await fetch('http://127.0.0.1:8080/train/', {
    // const response = await fetch('https://builder-demo-production-f269.up.railway.app/train/', {
    // const response = await fetch('https://deep-dive-into-ai-backend.onrender.com/train/', {
    const response = await fetch('http://15.134.33.92:8000/train/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }, 
        body: JSON.stringify( inputs ),
    });

    console.log("Training process completed. Processing response...");

    if (!response.ok) {
        throw new Error(`Error: ${await response.text()}`)
    }

    console.log("Response OK. Parsing JSON...");

    const responseJSON = await response.json();
    
    const body = JSON.stringify(responseJSON);
    const headers = {
        'Content-Type': 'application/json',
    };
    return new Response(body, { headers });
    
}) satisfies RequestHandler;
