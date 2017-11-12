//config the firebase
var config = {
    apiKey: "AIzaSyD7ZyQKmzVxvsw_BZ0ThAI6RaCOPmAvwSM",
    authDomain: "webtest-w1wsn.firebaseapp.com",
    databaseURL: "https://webtest-w1wsn.firebaseio.com",
    projectId: "webtest-w1wsn",
    storageBucket: "webtest-w1wsn.appspot.com",
    messagingSenderId: "1069173100525"
 };
 firebase.initializeApp(config);


 //create reference
var dbRefimg = firebase.database().ref().child('Img');
var ref = firebase.database().ref("users/ada");


ref.once("value")
  .then(function(snapshot) {
    var a = snapshot.exists();  // true
    var b = snapshot.child("name").exists(); // true
    var c = snapshot.child("name/first").exists(); // true
    var d = snapshot.child("name/middle").exists(); // false
  });


$('#save_button').click(function(){
	dbRefimg.set({
		imgURL:$('#imgURL').val(),
		imgName:$('#imgName').val()
	});
});

$('#clear_button').click(function(){
		$('#imgURL').val(''),
		$('#imgName').val('')
});


$('#load_button').click(function(){
	dbRefimg.on('value',function(snap){
		$('#imgURL').val(snap.child('imgURL').val());
		$('#imgName').val(snap.child('imgName').val());
	})
})

var fileButton = document.getElementById('fileButton');
var storageRef;
var file;
fileButton.addEventListener('change',function(e){
	//get file
	file = e.target.files[0];
	storageRef = firebase.storage().ref('img/'+file.name);
});

$('#submit').click(function(){
	storageRef.put(file);
})