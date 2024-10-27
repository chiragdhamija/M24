import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { AlertTriangle, CheckCircle } from 'lucide-react';

const AlertComponent = () => {
    const [alert, setAlert] = useState(null);

    useEffect(() => {
        const fetchAlertData = async () => {
            try {
                const response = await axios.get('http://localhost:5000/alert-data');
                // Assuming the response.data is either 0 or 1
                setAlert(response.data);
            } catch (error) {
                console.error('Error fetching alert data:', error);
            }
        };

        fetchAlertData();
    }, []); // Runs once when the component mounts

    return (
        <div className="p-4">
          {alert === 1 && (
            <div className="bg-red-600 text-white p-6 rounded-lg shadow-lg animate-pulse flex items-center gap-4 border-2 border-red-400">
              <AlertTriangle className="h-8 w-8" />
              <p className="text-2xl font-bold">Alert: Cyclone warning! Conditions met!</p>
            </div>
          )}
          
          {alert === 0 && (
            <div className="bg-green-600 text-white p-6 rounded-lg shadow-lg flex items-center gap-4 border-2 border-green-400">
              <CheckCircle className="h-8 w-8" />
              <p className="text-2xl font-bold">No cyclone is near. All is well!</p>
            </div>
          )}
        </div>
      );
};

export default AlertComponent;
