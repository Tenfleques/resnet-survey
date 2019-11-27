import React from 'react';

const Image = (props) => {
    return (
        <div className="col-md-10 col-lg-6">
            <img className="img-thumbnail border-0 img-fluid  float-right"  src={props.src} alt="" />
        </div>        
    );    
}
export default Image
  
  