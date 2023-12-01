using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Target : MonoBehaviour
{
    private void OnTriggerEnter(Collider other)
    {
        Debug.Log("Trigger: " + other.transform.name);
        if(other.transform.name == "Cube")
        {
            other.gameObject.GetComponent<Test>().notRecievedReward += 5;
            transform.position = new Vector3 (Random.Range(-5,5), 0, Random.Range(-5,5));
        }
    }
}
