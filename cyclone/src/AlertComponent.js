import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { AlertTriangle, CheckCircle } from 'lucide-react';
import "./alert.css"

const AlertComponent = () => {
    const [alert, setAlert] = useState(null);

    useEffect(() => {
        const fetchAlertData = async () => {
            try {
                const response = await axios.get('http://localhost:5000/alert-data');
                // Assuming the response.data is either 0 or 1
                // console.log(response.data);
                setAlert(response.data);
            } catch (error) {
                console.error('Error fetching alert data:', error);
            }
        };
        fetchAlertData();
    }, []); // Runs once when the component mounts

    return (
        <div className="p-4">
          {console.log(alert)}
          {alert === "1"&& (
            <div className="alert-warning">
              <AlertTriangle className="h-8 w-8" />
              <p className="text-2xl font-bold">Alert: Cyclone warning! Conditions met!</p>
            </div>
          )}
          
          {alert == 0 && (
            <div className="alert-success">
              <CheckCircle className="h-8 w-8" />
              <p className="text-2xl font-bold">No cyclone is near. All is well!</p>
            </div>
          )}
        </div>
      );
};

export default AlertComponent;
