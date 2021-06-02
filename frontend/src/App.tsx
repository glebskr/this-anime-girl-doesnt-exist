import React from 'react';
import Select from 'react-select';
import { capitalize, get, sample } from 'lodash';
import { Button } from '@material-ui/core';

import './App.css';

const EYE_COLORS = [
  'gray',
  'black',
  'orange',
  'pink',
  'yellow',
  'aqua',
  'purple',
  'green',
  'brown',
  'red',
  'blue',
];
const HAIR_COLORS = [
  'orange',
  'white',
  'aqua',
  'gray',
  'green',
  'red',
  'purple',
  'pink',
  'blue',
  'black',
  'brown',
  'blonde',
];

const eyeOptions = EYE_COLORS.map((color) => ({ value: color, label: capitalize(color) }));
const hairOptions = HAIR_COLORS.map((color) => ({ value: color, label: capitalize(color) }));

const App = () => {
  const [image, setImage] = React.useState('');
  const [hair, setHair] = React.useState('');
  const [eyes, setEyes] = React.useState('');
  React.useEffect(() => {
    fetch(`/anime?hair=${hair}&eyes=${eyes}`)
      .then((response) => {
        return response.blob();
      })
      .then((payload) => {
        setImage(URL.createObjectURL(payload));
      });
  }, [hair, eyes]);

  const onButtonClick = () => {
    setEyes(sample(EYE_COLORS) as string);
    setHair(sample(HAIR_COLORS) as string);
  };

  return (
    <div className="App">
      <h1>This anime does not exist</h1>
      <div>AI powered anime girls generator</div>
      <div>
        <img className="anime-img" src={image} />
      </div>
      <div className="selectors-conatiner">
        <div>
          <span className="selector-header">Hair color</span>
          <Select<{ value: string; label: string }, false>
            options={hairOptions}
            onChange={(value: { value: string; label: string } | null) =>
              setHair(get(value, 'value', ''))
            }
          />
        </div>
        <div>
          <span className="selector-header">Eyes color</span>
          <Select<{ value: string; label: string }, false>
            options={eyeOptions}
            onChange={(value: { value: string; label: string } | null) =>
              setEyes(get(value, 'value', ''))
            }
          />
        </div>
      </div>
      <Button variant="contained" color="primary" onClick={onButtonClick}>
        Random
      </Button>
    </div>
  );
};

export default App;
