import React from 'react';

function imageExists(image_url){
    var http = new XMLHttpRequest();
    http.open('HEAD', image_url, false);
    http.send();

    return http.status !== 404;

}
function hideImageError(id) {
    let classlist = document.getElementById(id).classList 
    if (classlist.contains("d-none"))
        return 
    document.getElementById(id).classList += " d-none";
}

function revealOnSuccess(id){
    document.getElementById(id).classList.remove("invisible")
    document.getElementById(id).classList.remove("d-none")
}

const Image = (props) => {
    let id = "image-block" + props.name.replace(/[^\w\s\_]/g, "").replace(/\s+| /g, "") + Math.round(Math.random()*1000)
    return (
        <div id={id} className="col-12 invisible">
            <h5>{props.name}</h5>
            <img className="img-thumbnail border-0 img-fluid  float-right"  src={props.src} alt="" onLoad={() => {revealOnSuccess(id) }} onError={()=> hideImageError(id)} />
        </div>        
    );    
}
export default Image
  
  