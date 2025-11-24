import type { RequestHandler } from './$types';

export const POST: RequestHandler = (async ({request}) => {
    const inputs = await request.json();

    // const response = await fetch('http://127.0.0.1:8080/train/', {
    // const response = await fetch('https://builder-demo-production-f269.up.railway.app/train/', {
    const response = await fetch('https://deep-dive-into-ai-backend.onrender.com/train/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }, 
        body: JSON.stringify( inputs ),
    });

    if (!response.ok) {
        throw new Error(`Error: ${await response.text()}`)
    }

    const responseJSON = await response.json();
    
    const body = JSON.stringify(responseJSON);
    const headers = {
        'Content-Type': 'application/json',
    };
    return new Response(body, { headers });
    
}) satisfies RequestHandler;
