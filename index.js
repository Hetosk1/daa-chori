const express = require('express')

const app = express()

app.get('/', (request, response) => {
	return response.json({
		"message": "tari ben no piko"
	});

})

app.get('/linear', (request, response) => {
	return response.json({
		"code": `		
def linear_search(arr, key):
    for i in range(0,  len(arr)+1):
        if arr[i] == key:
            return i+1
    return -1


print(linear_search([1, 2, 3, 4, 5], 3))
		`
	})
})

app.listen(5000, () => {
	console.log("chalu thai gyu");
})
