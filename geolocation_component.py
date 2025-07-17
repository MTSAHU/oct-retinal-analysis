# geolocation_component.py
import streamlit.components.v1 as components

def get_geolocation():
    """
    Creates a Streamlit component that uses JavaScript to get the user's geolocation.
    Returns latitude and longitude if successful, otherwise None.
    """
    return_value = components.html(
        """
        <script>
            var lat = null;
            var lon = null;
            var timeoutId = null; // To clear timeout if location found

            function sendLocationToStreamlit(latitude, longitude) {
                // Ensure the parent element exists before trying to send
                if (window.parent && window.parent.postMessage) {
                    window.parent.postMessage({
                        streamlit: {
                            setComponentValue: {
                                value: { lat: latitude, lon: longitude }
                            }
                        }
                    }, '*');
                }
            }

            if ("geolocation" in navigator) {
                navigator.geolocation.getCurrentPosition(
                    function(position) {
                        lat = position.coords.latitude;
                        lon = position.coords.longitude;
                        sendLocationToStreamlit(lat, lon);
                        if (timeoutId) clearTimeout(timeoutId); // Clear timeout if location found
                    },
                    function(error) {
                        console.error("Geolocation error:", error);
                        // Send null to indicate failure
                        sendLocationToStreamlit(null, null);
                        if (timeoutId) clearTimeout(timeoutId); // Clear timeout even on error
                    },
                    {
                        enableHighAccuracy: true,
                        timeout: 10000, // 10 seconds
                        maximumAge: 0
                    }
                );
            } else {
                console.log("Geolocation is not supported by this browser.");
                sendLocationToStreamlit(null, null); // Indicate not supported
            }

            // Set a timeout to send null if geolocation takes too long or fails silently
            timeoutId = setTimeout(function() {
                if (lat === null && lon === null) {
                    sendLocationToStreamlit(null, null);
                    console.warn("Geolocation timed out or failed to return a value.");
                }
            }, 12000); // Give it a bit more time than the getCurrentPosition timeout
        </script>
        """,
        height=1, # This component is invisible
        width=1,
        scrolling=False # Remove the 'key' argument here!
    )
    return return_value