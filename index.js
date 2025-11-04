const express = require('express')

const app = express()

app.get('/', (request, response) => {
	return response.json({
		"message": "tari ben no piko"
	});

})

app.listen(5000, () => {
	console.log("chalu thai gyu");
})
